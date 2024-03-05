import os

import torch
from loguru import logger  # noqa: F401
from model.utility import MessagePassingLayer, unpool_edge
from torch import nn
from torch.nn import SELU, LayerNorm, Linear, Sequential
from torch_geometric.data import Batch


class Decoder(nn.Module):
    def __init__(self, args, m_ids, m_gs, e_s):
        super(Decoder, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.latent_edge_dim = m_gs[-1].shape[-1]
        self.max_hidden_dim = args.hidden_dim * 2**args.ae_layers
        # Pre computed node mask and edge_mask from bi-stride pooling
        self.m_ids, self.m_gs, self.e_x = m_ids, m_gs, e_s
        self.ae_layers = args.ae_layers
        self.n = args.n_nodes
        self.layers = nn.ModuleList()
        self.out_feature_dim = args.out_feature_dim
        self.latent_vec_dim = len(m_ids[-1])
        self.mpl_bottom = MessagePassingLayer(
            hidden_dim=args.latent_dim, latent_dim=self.max_hidden_dim, args=args
        )

        self.linear_up_mlp = Sequential(
            Linear(1, 64), SELU(), Linear(64, self.latent_vec_dim)
        )

        for i in range(self.ae_layers):
            if i == self.ae_layers - 1:
                up_nodes = self.n
            else:
                up_nodes = len(m_ids[args.ae_layers - i - 2])
            self.layers.append(
                Res_up(
                    channel_in=self.max_hidden_dim // 2**i,
                    channel_out=self.max_hidden_dim // 2 ** (i + 1),
                    args=args,
                    m_id=m_ids[args.ae_layers - i - 1],
                    m_g=m_gs[args.ae_layers - i - 1],
                    e_idx=e_s[args.ae_layers - i - 1],
                    up_nodes=up_nodes,
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
        self.out_edge_encoder = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            SELU(),
            Linear(self.hidden_dim // 2, 3),
            LayerNorm(3),
        )

    def from_latent_vec(self, z_x):
        x = self.linear_up_mlp(z_x)
        x = self.batch_to_sparse(x)
        return x

    def batch_to_sparse(self, z):
        z = z.transpose(1, 2)
        z = z.contiguous().view(-1, self.args.latent_dim)
        return z

    def load_and_populate_b_data(self, z_x, batch_size):
        z_x = z_x.view(batch_size, -1, self.args.latent_dim)
        PATH = os.path.join(self.args.graph_structure_aelayers_dir, "b_data.pt")
        b_data = torch.load(PATH).to(self.args.device)
        b_lst = []
        for i in range(batch_size):
            data = b_data.clone()
            # z_x should be [4, 241, 128]
            data.x = z_x[i]
            b_lst.append(data)
        return Batch.from_data_list(b_lst)

    def forward(self, z):
        batch_size = z.shape[0]
        z_x = self.from_latent_vec(z)

        b_data = self.load_and_populate_b_data(z_x, batch_size)
        b_data = self.mpl_bottom(b_data)
        for i in range(self.ae_layers):
            b_data = self.layers[i](b_data)
        b_data = self.final_layer(b_data)
        b_data.x = self.out_node_decoder(b_data.x)
        b_data.edge_attr = self.out_edge_encoder(b_data.edge_attr)
        return b_data


""" if not os.path.isfile(b_data_path):
       torch.save(b_data[0], b_data_path)
 b_data = torch.load(b_data_path)
 batch_size = len(np.unique(b_data.batch))
 b_lst = []
 for i in range(batch_size):
    data = b_data.clone()
    data.x = z[i].squeeze()
 return Batch.from_data_list(b_lst)"""


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


class Res_up(nn.Module):
    def __init__(self, channel_in, channel_out, args, m_id, m_g, e_idx, up_nodes):
        super(Res_up, self).__init__()
        self.m_id = m_id
        self.m_g = m_g
        self.e_idx = e_idx
        self.args = args
        self.up_nodes = up_nodes
        self.mpl1 = MessagePassingLayer(channel_in, channel_out // 2, args)
        self.mpl2 = MessagePassingLayer(channel_out // 2, channel_out, args)
        self.mpl_skip = MessagePassingLayer(channel_in, channel_out, args)
        self.unpool = Unpool()
        self.act1 = nn.SELU()

    def _bi_up_pool_batch(self, b_data):
        b_lst = b_data.to_data_list()
        batch_lst = []
        g, mask = self.m_g, self.m_id
        if not torch.is_tensor(g):
            g = torch.tensor(g)
        for idx, data in enumerate(b_lst):
            data.x = self.unpool(data.x, self.up_nodes, mask)
            data.weights = self.unpool(data.weights, self.up_nodes, mask)
            data.edge_index, data.edge_attr = unpool_edge(
                g, data.edge_attr, self.e_idx, self.args
            )
            batch_lst.append(data)
        b_data = Batch.from_data_list(batch_lst).to(self.args.device)
        return b_data

    def forward(self, b_data):
        b_skip = self.mpl_skip(self._bi_up_pool_batch(b_data.clone()))
        b_data = self.mpl1(b_data)
        b_data = self._bi_up_pool_batch(b_data)
        b_data = self.mpl2(b_data)
        b_data.x = self.act1(b_data.x + b_skip.x)
        return b_data
