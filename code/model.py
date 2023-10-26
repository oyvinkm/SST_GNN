"""
    This file contains the classes used to build our Multi Scale Auto Encoder GNN.
"""

import torch
from torch import nn
from torch.nn import LayerNorm, Linear, ReLU, Sequential
from torch_geometric.nn.conv import GraphConv, MessagePassing
from torch_geometric.nn.pool import ASAPooling, SAGPooling, TopKPooling
from torch_geometric.utils import degree
from torch_scatter import scatter


class MultiScaleAutoEncoder(nn.Module):
    """
    Multiscale Auto Encoder consist of n_layer of Message Passing Layers (MPL) with
    pooling and unpooling operations in between in order to obtain a coarse latent
    representation of a graph. Uses an Multilayer Perceptron (MLP) to compute node and
    edge features.
    Encode: G_0 -> MLP -> MPL -> TopKPool ... MPL -> G_l
    Decode: G_l -> MPL -> Unpool .... -> MPL -> MLP -> G'_0
    """

    def __init__(self, args):
        super().__init__()
        self.in_dim_node = args.in_dim_node
        self.in_dim_edge = args.in_dim_edge
        if args.out_feature_dim is None:
            self.out_feature_dim = args.in_dim_node
        else:
            self.out_feature_dim = args.out_feature_dim
        self.hidden_dim = args.hidden_dim
        self.ae_layers = args.ae_layers
        if args.ae_ratio is None:
            self.ae_ratio = 0.5
        else:
            self.ae_ratio = args.ae_ratio
        if args.latent_dim is None:
            self.latent_dim = args.hidden_dim
        else:
            self.latent_dim = args.latent_dim
        self.args = args
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.down_pool = nn.ModuleList()
        self.up_pool = nn.ModuleList()

        for _ in range(self.ae_layers):
            self.down_layers.append(MessagePassingLayer(args=self.args))
            self.up_layers.append(MessagePassingLayer(args=self.args))
            self.down_pool.append(TopKPooling(self.hidden_dim, self.ae_ratio))
            self.up_pool.append(Unpool())

        self.bottom_layer = MessagePassingLayer(args=self.args)

        self.final_layer = MessagePassingLayer(args=self.args)

        self.node_encoder = Sequential(
            Linear(self.in_dim_node, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, self.latent_dim),
            LayerNorm(self.latent_dim),
        )
        self.out_node_decoder = Sequential(
            Linear(self.latent_dim, self.latent_dim // 2),
            ReLU(),
            Linear(self.latent_dim // 2, self.out_feature_dim),
            LayerNorm(self.out_feature_dim),
        )

        self.edge_encoder = Sequential(
            Linear(self.in_dim_edge, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, self.hidden_dim),
            LayerNorm(self.hidden_dim),
        )

    def forward(self, b_data):
        """Forward loop, first encoder, then bottom layer, then decoder"""
        x = b_data.x
        edge_attr = b_data.edge_attr
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        in_edge_attr = b_data.edge_attr
        down_gs = []
        down_outs = []
        perms = []
        ws = []
        bs = []
        b_data.x = x
        b_data.edge_attr = edge_attr
        for i in range(self.ae_layers):
            b_data = self.down_layers[i](b_data)
            batch = b_data.batch
            ws.append(b_data.weights)
            bs.append(batch)

            down_outs.append(b_data.x)
            down_gs.append(b_data.edge_index)
            x, edge_index, edge_attr, batch, perm, _ = self.down_pool[i](
                b_data.x, b_data.edge_index, b_data.edge_attr, batch
            )
            b_data.x = x
            b_data.edge_index = edge_index
            b_data.edge_attr = edge_attr
            b_data.batch = batch
            b_data.weights = b_data.weights[perm]
            perms.append(perm)

        # Do the final MMP before we arrive at G_L
        b_data = self.bottom_layer(b_data)
        z = b_data.clone()
        for i in range(self.ae_layers):
            up_idx = self.ae_layers - i - 1
            x = self.up_layers[i](b_data)
            x = self.up_pool[i](b_data.x, down_outs[up_idx].shape[0], perms[up_idx])
            b_data.weights = ws[up_idx]
            b_data.batch = bs[up_idx]
            b_data.x = x
            b_data.edge_index = down_gs[up_idx]

        x = self.final_layer(b_data)
        x = self.out_node_decoder(b_data.x)
        b_data.x = x
        b_data.edge_attr = in_edge_attr

        return b_data, z  # , edge_attr, edge_index
    
            


class MessagePassingLayer(torch.nn.Module):
    """
    Kinda like a U-Net but with Message Passing Blocks.
    """

    def __init__(self, args):
        super(MessagePassingLayer, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.l_n = args.mpl_layers
        self.args = args
        if args.latent_dim is None:
            self.latent_dim = args.hidden_dim
        else:
            self.latent_dim = args.latent_dim
        self.num_blocks = args.num_blocks
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.bottom_gmp = MessagePassingBlock(hidden_dim=self.latent_dim, args=args)
        self.edge_conv = WeightedEdgeConv()
        self.pools = nn.ModuleList()
        if self.args.mpl_ratio is None:
            self.ae_ratio = 0.5
        else:
            self.ae_ratio = self.args.mpl_ratio

        for _ in range(self.l_n):
            self.down_gmps.append(
                MessagePassingBlock(hidden_dim=self.latent_dim, args=args)
            )
            self.up_gmps.append(
                MessagePassingBlock(hidden_dim=self.latent_dim, args=args)
            )
            self.unpools.append(Unpool())
            if self.args.pool_strat == "ASA":
                self.pools.append(
                    ASAPooling(
                        in_channels=self.latent_dim, ratio=self.ae_ratio, GNN=GraphConv
                    )
                )
            else:
                self.pools.append(TopKPooling(self.hidden_dim, self.ae_ratio))

    def forward(self, b_data):
        """Forward pass through Message Passing Layer"""
        down_outs = []
        cts = []
        down_masks = []
        down_gs = []
        batches = []
        ws = []
        b_data.edge_weight = None
        edge_attr = b_data.edge_attr

        for i in range(self.l_n):
            h = b_data.x
            g = b_data.edge_index
            b_data.x = self.down_gmps[i](h, g)
            # record the infor before aggregation
            down_outs.append(h)
            down_gs.append(g)
            batches.append(b_data.batch)
            ws.append(b_data.weights)

            # aggregate then pooling
            # Calculates edge and node weigths
            ew, w = self.edge_conv.cal_ew(b_data.weights, g)
            b_data.weights = w
            # Does edge convolution on nodes with edge weigths
            b_data.x = self.edge_conv(h, g, ew)
            # Does edge convolution on position with edge weights
            cts.append(ew)
            if self.args.pool_strat == "ASA":
                x, edge_index, edge_weight, batch, index = self.pools[i](
                    b_data.x, b_data.edge_index, b_data.edge_weight, b_data.batch
                )
                down_masks.append(index)
                b_data.x = x
                b_data.edge_index = edge_index
                b_data.edge_weight = edge_weight
                b_data.batch = batch
                b_data.weights = b_data.weights[index]
            else:
                x, edge_index, edge_weight, batch, index, _ = self.pools[i](
                    b_data.x, b_data.edge_index, b_data.edge_weight, b_data.batch
                )
                down_masks.append(index)
                b_data.x = x
                b_data.edge_index = edge_index
                b_data.edge_weight = edge_weight
                b_data.batch = batch
                b_data.weights = b_data.weights[index]

        b_data.x = self.bottom_gmp(b_data.x, b_data.edge_index)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            h = self.unpools[i](
                b_data.x, down_outs[up_idx].shape[0], down_masks[up_idx]
            )
            tmp_g = down_gs[up_idx]
            h = self.edge_conv(h, tmp_g, cts[up_idx], aggragating=False)
            h = self.up_gmps[i](h, g)
            h = h.add(down_outs[up_idx])
            b_data.x = h
            b_data.edge_index = tmp_g
            b_data.batch = batches[up_idx]
            b_data.weights = ws[up_idx]
        b_data.edge_attr = edge_attr
        return b_data

    def _pooling_strategy(self):
        if self.args.pool_strat == "ASA":
            pool = ASAPooling
        elif self.args.pool_strat == "SAG":
            pool = SAGPooling
        else:
            pool = TopKPooling
        return pool


class WeightedEdgeConv(MessagePassing):
    """
    Weighted Edge Convolution used for pooling and unpooling.
    """

    def __init__(self):
        super().__init__(aggr="add", flow="target_to_source")

    def forward(self, x, g, ew, aggragating=True):
        # aggregating: False means returning
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            weighted_info = x[:, i] if aggragating else x[:, j]
        elif len(x.shape) == 2:
            weighted_info = x[i] if aggragating else x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        weighted_info *= ew.unsqueeze(-1)
        target_index = j if aggragating else i
        aggr_out = scatter(
            weighted_info, target_index, dim=-2, dim_size=x.shape[-2], reduce="sum"
        )
        return aggr_out

    @torch.no_grad()
    def cal_ew(self, w, g):
        deg = degree(g[0], dtype=torch.float, num_nodes=w.shape[0])
        normed_w = w.squeeze(-1) / deg
        i = g[0]
        j = g[1]
        w_to_send = normed_w[i]
        eps = 1e-12
        aggr_w = (
            scatter(w_to_send, j, dim=-1, dim_size=normed_w.size(0), reduce="sum") + eps
        )
        ec = w_to_send / aggr_w[j]
        return ec, aggr_w


class MessagePassingBlock(torch.nn.Module):
    """
    Just combines n number of message passing layers
    """

    def __init__(self, hidden_dim, args, num_blocks=None):
        super(MessagePassingBlock, self).__init__()
        if num_blocks is None:
            self.num_blocks = args.num_blocks
        else:
            self.num_blocks = num_blocks

        self.processor = nn.ModuleList()
        assert self.num_blocks >= 1, "Number of message passing layers is not >=1"

        processor_layer = self.build_processor_model()
        for _ in range(self.num_blocks):
            self.processor.append(processor_layer(hidden_dim, hidden_dim))

    def build_processor_model(self):
        return GCNConv

    def forward(self, x, edge_index):
        # Step 1: encode node/edge features into latent node/edge embeddings
        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_blocks):
            x = self.processor[i](x, edge_index)
        return x


class Unpool(nn.Module):
    """
    Fills an empty array
    """

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, h, pre_node_num, idx):
        new_h = h.new_zeros([pre_node_num, h.shape[-1]])
        new_h[idx] = h
        return new_h


class GCNConv(MessagePassing):
    """
    Classic MessagePassing/Convolution
    """

    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr="add")  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        # print(edge_index[0].shape)
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [num_edges, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [num_nodes, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out
