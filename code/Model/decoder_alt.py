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
from utility import MessagePassingLayer, GCNConv, WeightedEdgeConv, MessagePassingBlock, MessagePassingEdgeConv

class Decoder(nn.Module):
    def __init__(self, args, m_ids, m_gs):
        super(Decoder, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.max_hidden_dim = args.hidden_dim * 2 ** args.ae_layers
        self.m_ids, self.m_gs = m_ids, m_gs
        self.ae_layers = args.ae_layers
        self.n = args.n_nodes
        self.layers = nn.ModuleList()
        self.out_feature_dim = args.out_feature_dim
        self.latent_vec_dim = self.latent_vec_dim = len(m_ids[-1])
        self.mpl_bottom = MessagePassingEdgeConv(channel_in = args.latent_dim, 
                                              channel_out=self.max_hidden_dim, 
                                              args=args)
        self.linear_up_mlp = Sequential(Linear(1, 64),
                                    LeakyReLU(),
                                    Linear(64, self.latent_vec_dim))

        for i in range(self.ae_layers):
            if i == self.ae_layers - 1:
                up_nodes = self.n
            else:
                up_nodes = len(m_ids[args.ae_layers - i- 2])
            self.layers.append(Res_up(channel_in=self.max_hidden_dim // 2 ** i,
                                      channel_out=self.max_hidden_dim // 2 ** (i+1),
                                      args = args,
                                      m_id = m_ids[args.ae_layers - i - 1],
                                      m_g = m_gs[args.ae_layers - i - 1],
                                      up_nodes = up_nodes)
                                      )

        self.final_layer = MessagePassingEdgeConv(channel_in = self.hidden_dim, channel_out = self.hidden_dim, args=self.args)
        self.out_node_decoder = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            LeakyReLU(),
            Linear(self.hidden_dim // 2, self.out_feature_dim),
            LayerNorm(self.out_feature_dim),
        )

    def from_latent_vec(self, z_x):
        b_data = self.linear_up_mlp(z_x)
        b_data = self.batch_to_sparse(b_data)
        return b_data

    def batch_to_sparse(self, z):
        z = z.transpose(1,2)
        z = z.contiguous().view(-1, self.args.latent_dim)
        return z

    def forward(self, b_data, z):
        z_x = self.from_latent_vec(z)
        b_data.x = z_x
        b_data = self.mpl_bottom(b_data)
        for i in range(self.ae_layers):
           b_data = self.layers[i](b_data)
        b_data = self.final_layer(b_data) #
        b_data.x = self.out_node_decoder(b_data.x) #
        return b_data
    
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
    def __init__(self, channel_in, channel_out, args, m_id, m_g, up_nodes):
        super(Res_up, self).__init__()
        self.m_id = m_id
        self.m_g = m_g
        self.args = args
        self.up_nodes = up_nodes
        self.mpl1 = MessagePassingEdgeConv(channel_in=channel_in, channel_out=channel_out//2, args = args)
        self.mpl2 = MessagePassingEdgeConv(channel_in=channel_out//2, channel_out = channel_out, args = args)
        self.mpl_skip = MessagePassingEdgeConv(channel_in=channel_in, channel_out=channel_out, args = args)
        self.unpool = Unpool()
        self.act1 = nn.LeakyReLU()
    
    def _bi_up_pool_batch(self, b_data):
        b_lst = b_data.to_data_list()
        batch_lst = []
        g, mask = self.m_g, self.m_id
        if not torch.is_tensor(g):
            g = torch.tensor(g)
        for idx, data in enumerate(b_lst):
            data.x = self.unpool(data.x, self.up_nodes, mask)
            data.weights = self.unpool(data.weights, self.up_nodes, mask)
            data.edge_index = g
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