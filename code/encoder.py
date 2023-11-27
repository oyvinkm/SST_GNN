import numpy as np
import torch
from torch import nn
from torch.nn import LayerNorm, Linear, ReLU, Sequential, LeakyReLU
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import GraphConv, MessagePassing
from torch_geometric.nn.pool import ASAPooling, SAGPooling, TopKPooling
from torch_geometric.utils import degree
from torch_scatter import scatter
from loguru import logger

class Encoder(nn.Module):
    def __init__(self, args, m_ids, m_gs):
        super(Encoder, self).__init__()
        self.args = args
        self.m_ids, self.m_gs = m_ids, m_gs
        self.ae_layers = args.ae_layers
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.in_dim_node = args.in_dim_node
        self.latent_vec_dim = len(m_ids[-1])
        self.down_layers = nn.ModuleList()
        self.down_pool = nn.ModuleList()
        self.b = args.batch_size


        self.node_encoder = Sequential(
            Linear(self.in_dim_node, self.hidden_dim),
            LeakyReLU(),
            Linear(self.hidden_dim, self.hidden_dim),
            LayerNorm(self.hidden_dim),
        )
        
        for i in range(self.ae_layers):
            if (i == self.ae_layers - 1):
                self.down_layers.append(
                    MessagePassingLayer(hidden_dim = self.hidden_dim *2, 
                                latent_dim = self.latent_dim, 
                                args=self.args)
                                )
            else:
                self.down_layers.append(
                    MessagePassingLayer(hidden_dim = self.hidden_dim, 
                                        latent_dim=self.hidden_dim, 
                                        args=self.args)
                                        )
            if self.args.ae_pool_strat == "ASA":
                self.down_pool.append(
                    self.pool(
                        in_channels=self.hidden_dim, ratio=self.ae_ratio, GNN=GraphConv
                    )
                )
            else:
                self.down_pool.append(
                    self.pool(self.hidden_dim, self.ae_ratio)
                )
            self.residual_MP = MessagePassingBlock(self.latent_dim, self.hidden_dim, self.args)
        self.bottom_layer = MessagePassingLayer(hidden_dim = self.latent_dim, latent_dim = self.latent_dim, args=self.args, bottom=True)
        self.mlp_logvar = Sequential(Linear(self.latent_vec_dim, 64),
                        LeakyReLU(),
                        Linear(64, 1))

        self.mlp_mu = Sequential(Linear(self.latent_vec_dim, 64),
                        LeakyReLU(),
                        Linear(64, 1))

        self.linear_up_mlp = Sequential(Linear(1, 64),
                                    LeakyReLU(),
                                    Linear(64, self.latent_vec_dim))

    def forward(self, b_data, Train = True):
        x = b_data.x
        b_data.x = self.node_encoder(x)
        self.b = len(torch.unique(b_data.batch))
        res = self._map_res(b_data.clone().to_data_list()) #
        for i in range(self.ae_layers): #
            if i == self.ae_layers - 1 and self.residual: #
                b_data = self._residual_connection(b_data, res) #
            b_data = self.down_layers[i](b_data) #
            b_data = self._bi_pool_batch(b_data, i) #
        b_data = self.bottom_layer(b_data) #

        if Train:
            mu = self.to_latent_vec(b_data, self.mlp_mu)
            log_var = self.to_latent_vec(b_data, self.mlp_logvar)
            z = self.sample(mu, log_var)
            kl = torch.mean(-0.5 * torch.sum(1+log_var-mu**2-log_var.exp(), dim=1), dim=0)

        else:
            z = self.to_latent_vec(b_data, self.mlp_mu)
            kl = None

        return kl, z, b_data

    def _map_res(self, b_lst):
        res = [data.x for data in b_lst]
        for id in self.m_ids[:-1]:
            res = [x[id] for x in res]
        return res

    def _residual_connection(self, b_data, res):
        p_lst = b_data.to_data_list()
        for idx, data in enumerate(p_lst):
            data.x = torch.cat((data.x, res[idx]), dim = 1)
        return Batch.from_data_list(p_lst).to(self.args.device)

    def _bi_pool_batch(self, b_data, i):
        b_lst = Batch.to_data_list(b_data)
        data_lst = []
        if not torch.is_tensor(self.m_gs[i+1]):
            g = torch.tensor(self.m_gs[i+1])
        mask = self.m_ids[i]
        for idx, data in enumerate(b_lst):
            data.x = data.x[mask]
            data.weights = data.weights[mask]
            data.edge_index = g
            data_lst.append(data)
        return Batch.from_data_list(data_lst).to(self.args.device)

    def to_latent_vec(self, b_data, mpl):
        b_data = self.batch_to_dense_transpose(b_data)
        z_x = mpl(b_data)
        return z_x

    def sample(self, mu, logvar):
        """Shamelessly stolen from https://github.com/julschoen/Latent-Space-Exploration-CT/blob/main/Models/VAE.py"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


class MessagePassingLayer(torch.nn.Module):
    """
    Kinda like a U-Net but with Message Passing Blocks.
    The Multiscale Autoencoder consists of multiple of these
    """

    def __init__(self, hidden_dim, latent_dim, args, bottom = False, first_up = False):
        super(MessagePassingLayer, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.l_n = args.mpl_layers
        self.args = args
        self.bottom = bottom
        self.first_up = first_up
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        """ if args.latent_dim is not None and (bottom or  first_up):
            if first_up:
                self.hidden_dim = args.latent_dim
                self.latent_dim = args.hidden_dim
            else:
                self.latent_dim = args.latent_dim
        else:
            self.latent_dim = args.hidden_dim """
        self.num_blocks = args.num_blocks
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.bottom_gmp = MessagePassingBlock(hidden_dim=self.latent_dim, latent_dim=self.latent_dim, args=args)
        self.edge_conv = WeightedEdgeConv()
        self.pools = nn.ModuleList()
        if self.args.mpl_ratio is None:
            self.mpl_ratio = 0.5
        else:
            self.mpl_ratio = self.args.mpl_ratio

        for i in range(self.l_n):
            if i == 0:
                self.down_gmps.append(
                        MessagePassingBlock(hidden_dim=self.hidden_dim, latent_dim = self.latent_dim, args=args)
                    )
            else:
                self.down_gmps.append(
                       MessagePassingBlock(hidden_dim=self.latent_dim, latent_dim = self.latent_dim, args=args)
                    )
            self.up_gmps.append(
                MessagePassingBlock(hidden_dim=self.latent_dim, latent_dim=self.latent_dim, args=args)
            )
            self.unpools.append(Unpool())
            if self.args.pool_strat == "ASA":
                self.pools.append(
                    self.pool(
                        in_channels=self.latent_dim, ratio=self.mpl_ratio, GNN=GraphConv
                    )
                )
            else:
                self.pools.append(
                    self.pool(self.latent_dim, self.mpl_ratio)
                )

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
        b_data.weights = b_data.x.new_ones((b_data.x.shape[-2], 1)) if b_data.weights is None else b_data.weights
        for i in range(self.l_n):
            h = b_data.x
            g = b_data.edge_index
            h = self.down_gmps[i](h, g)
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

class MessagePassingBlock(torch.nn.Module):
    """
    Just combines n number of message passing layers
    """

    def __init__(self, hidden_dim, latent_dim, args, num_blocks=None):
        super(MessagePassingBlock, self).__init__()
        if num_blocks is None:
            self.num_blocks = args.num_blocks
        else:
            self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.processor = nn.ModuleList()
        assert self.num_blocks >= 1, "Number of message passing layers is not >=1"
        
        processor_layer = self.build_processor_model()
        for i in range(self.num_blocks):
            if i == 0:
                self.processor.append(processor_layer(self.hidden_dim, self.latent_dim))
            else:
                self.processor.append(processor_layer(self.latent_dim, self.latent_dim))

    def build_processor_model(self):
        return GCNConv

    def forward(self, x, edge_index):
        # Step 1: encode node/edge features into latent node/edge embeddings
        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_blocks):
            x = self.processor[i](x, edge_index)
        return x