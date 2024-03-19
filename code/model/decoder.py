import torch
from loguru import logger
from model.utility import MessagePassingLayer, Unpool
from torch import nn
from torch.nn import SELU, LayerNorm, Linear, Sequential
from torch_geometric.data import Batch
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.unpool import knn_interpolate


class Decoder(nn.Module):
    def __init__(self, args, m_ids, m_gs, e_s, m_pos, graph_placeholder):
        """
        Initializes the Decoder class.

        Args:
            args: The arguments for the model.
            m_ids: The node IDs.
            m_gs: The node graphs.
            e_s: The edge indices.
            m_pos: The node positions.
            graph_placeholder: The graph placeholder.
        """
        super(Decoder, self).__init__()
        self.args = args
        self.graph_placeholder = graph_placeholder
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.max_hidden_dim = args.hidden_dim * 2**args.ae_layers
        # Pre computed node mask and edge_mask from bi-stride pooling
        self.m_ids, self.m_gs, self.e_s = m_ids, m_gs, e_s
        self.ae_layers = args.ae_layers
        self.n = args.n_nodes
        self.layers = nn.ModuleList()
        self.out_feature_dim = args.out_feature_dim
        self.latent_node_dim = args.max_latent_nodes
        self.latent_edge_dim = args.max_latent_edges
        # 1 x 128 -> 1 x |V|
        self.up_mlp = Linear(self.latent_dim, self.latent_node_dim)

        # |V| x 1 -> |V| -> H
        self.latent_up_mlp = Sequential(
            Linear(1, self.latent_dim // 2),
            SELU(),
            Linear(self.latent_dim // 2, self.latent_dim),
        )
        self.mpl_bottom = MessagePassingLayer(
            hidden_dim=args.latent_dim, latent_dim=self.max_hidden_dim, args=args
        )

        for i in range(self.ae_layers):
            up_idx = args.ae_layers - i - 1
            if i == self.ae_layers - 1:
                up_nodes = self.n
            else:
                up_nodes = m_ids[up_idx - 1]
            self.layers.append(
                Res_up(
                    channel_in=self.max_hidden_dim // 2**i,
                    channel_out=self.max_hidden_dim // 2 ** (i + 1),
                    args=args,
                    m_id=m_ids[up_idx],
                    m_g=m_gs[up_idx],
                    e_idx=e_s[up_idx],
                    up_nodes=up_nodes,
                    m_pos_new=m_pos[up_idx],
                )
            )

        self.final_layer = MessagePassingLayer(
            hidden_dim=self.hidden_dim, latent_dim=self.hidden_dim, args=self.args
        )
        self.out_node_decoder = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            SELU(),
            Linear(self.hidden_dim // 2, self.out_feature_dim),
            LayerNorm(self.out_feature_dim),
        )

    def construct_batch(self, latent_vec):
        b_lst = []
        for z, t in latent_vec:
            graph = self.graph_placeholder[t].clone()
            node_mask = self.m_ids[-1][t]
            graph.x = z[: len(node_mask)]
            b_lst.append(graph)
        return Batch.from_data_list(b_lst).to(self.args.device)

    def forward(self, latent_vec):
        latent_vec.z = self.up_mlp(latent_vec.z).transpose(1, 2)
        latent_vec.z = self.latent_up_mlp(latent_vec.z)
        # Should be shape (B, |V|_max, latent_dim):
        b_data = self.construct_batch(latent_vec)
        b_data = self.mpl_bottom(b_data)
        for i in range(self.ae_layers):
            b_data = self.layers[i](b_data)
            if torch.any(torch.isnan(b_data.x)):
                logger.error(f"something is nan in decoder path no {i}")
                exit()
        b_data = self.final_layer(b_data)
        b_data.x = self.out_node_decoder(b_data.x)
        return b_data


class Res_up(nn.Module):
    def __init__(
        self, channel_in, channel_out, args, m_id, m_g, e_idx, up_nodes, m_pos_new
    ):
        """
        Initialize the Res_up class, Message Passing layers + upsampling with residual connections
        from layer l -> l-1.

        Args:
            channel_in (int): Number of input channels.
            channel_out (int): Number of output channels.
            args: Additional arguments.
            m_id: Identifier of the model.
            m_g: Model graph.
            e_idx: Edge index.
            up_nodes: Up nodes.
            m_pos_new: New model position.

        Returns:
            None
        """
        super(Res_up, self).__init__()
        self.m_id = m_id
        self.m_g = m_g
        self.e_idx = e_idx
        self.args = args
        self.up_nodes = up_nodes
        self.m_pos_new = m_pos_new
        self.mpl1 = MessagePassingLayer(channel_in, channel_out // 2, args)
        self.mpl2 = MessagePassingLayer(channel_out // 2, channel_out, args)
        self.mpl_skip = MessagePassingLayer(channel_in, channel_out, args)
        self.unpool = Unpool()
        self.act1 = SELU()
        self.act2 = SELU()
        self.bn_nodes = BatchNorm(in_channels=channel_out)

    def _bi_up_pool_batch(self, b_data):
        b_lst = b_data.to_data_list()
        batch_lst = []
        for idx, data in enumerate(b_lst):
            g, mask = self.m_g[data.trajectory], self.m_id[data.trajectory]
            up_nodes = (
                self.up_nodes
                if isinstance(self.up_nodes, int)
                else len(self.up_nodes[data.trajectory])
            )
            m_pos = self.m_pos_new[data.trajectory].to(self.args.device)

            data.x = knn_interpolate(data.x, data.mesh_pos, m_pos)
            data.mesh_pos = m_pos
            data.weights = self.unpool(data.weights, up_nodes, mask)
            data.edge_index = g
            batch_lst.append(data)
        b_data = Batch.from_data_list(batch_lst).to(self.args.device)
        return b_data

    def forward(self, b_data):
        b_skip = self.mpl_skip(self._bi_up_pool_batch(b_data.clone()))
        b_data = self.mpl1(b_data)
        b_data = self._bi_up_pool_batch(b_data)
        b_data = self.mpl2(b_data)
        b_data.x = b_data.x + b_skip.x
        if self.args.batch_norm:
            b_data.x = self.bn_nodes(b_data.x)
        b_data.x = self.act1(b_data.x)
        return b_data
