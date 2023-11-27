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

class Decoder(nn.Module):
    def __init__(self, args, m_ids, m_gs):
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.m_ids, self.m_gs = m_ids, m_gs
        self.ae_layers = args.ae_layers
        self.n = args.n_nodes
        self.up_layers = nn.ModuleList()
        self.up_pool = nn.ModuleList()
        self.out_feature_dim = args.out_feature_dim


        for i in range(self.ae_layers):
            if (i == self.ae_layers - 1):
                self.up_layers.append(
                    MessagePassingLayer(hidden_dim = self.hidden_dim * 2, 
                                        latent_dim = self.hidden_dim, 
                                        args=self.args)
                                        )
            if i == 0:
                self.up_layers.append(
                    MessagePassingLayer(hidden_dim = self.latent_dim, 
                                        latent_dim = self.hidden_dim, 
                                        args=self.args)
                                        )
            else:
                self.up_layers.append(
                    MessagePassingLayer(hidden_dim = self.hidden_dim, 
                                        latent_dim = self.hidden_dim, 
                                        args=self.args)
                                        )
            self.residual_MP = MessagePassingBlock(self.latent_dim, self.hidden_dim, self.args)
            self.up_pool.append(Unpool())
        self.residual_unpool = Unpool()
        self.final_layer = MessagePassingLayer(hidden_dim = self.hidden_dim, latent_dim = self.hidden_dim, args=self.args)

        self.out_node_decoder = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            LeakyReLU(),
            Linear(self.hidden_dim // 2, self.out_feature_dim),
            LayerNorm(self.out_feature_dim),
        )

    def forward(self, b_data, z):
        z_x = self.from_latent_vec(z)
        b_data = self.decode(b_data, z_x)
        b_data.x = z_x
        res_up = self._res_upsample(b_data, 0) #
        for i in range(self.ae_layers): #
            if i == self.ae_layers - 1: #
                b_data = self._residual_connection_up(b_data, res_up) #
            up_idx = self.ae_layers - i - 1 #
            b_data = self.up_layers[i](b_data) #
            b_lst = b_data.to_data_list() #
            batch_lst = [] # 
            g, mask = self.m_gs[up_idx], self.m_ids[up_idx] #
            up_nodes = len(self.m_ids[up_idx - 1]) if up_idx != 0  else self.n #
            if not torch.is_tensor(g):
                g = torch.tensor(g)
            for idx, data in enumerate(b_lst):
                data.x = self.up_pool[i](data.x, up_nodes, mask) #
                data.weights = self.up_pool[i](data.weights, up_nodes, mask) #
                data.edge_index = g #
                batch_lst.append(data) #
            b_data = Batch.from_data_list(batch_lst).to(self.args.device) #
        b_data = self.final_layer(b_data) #
        b_data.x = self.out_node_decoder(b_data.x) #
        return b_data
    
    def _res_upsample(self, b_data, i):
        up_nodes = len(self.m_ids[i])
        g = torch.tensor(self.m_gs[i + 1]).to(self.args.device)
        mask = self.m_ids[self.ae_layers - 1]
        b_lst = b_data.clone().to_data_list()
        unpooled = [self.residual_MP(self.residual_unpool(data.x, up_nodes, mask), g) for data in b_lst]
        return unpooled

    def _residual_connection_up(self, b_data, res_up):
        b_lst = b_data.clone().to_data_list()
        for i,d in enumerate(b_lst):
            d.x = torch.cat((d.x, res_up[i]), dim = 1)
        return  Batch.from_data_list(b_lst)

    def from_latent_vec(self, z_x):
        b_data = self.linear_up_mlp(z_x)
        b_data = self.batch_to_sparse(b_data)
        return b_data

    def batch_to_sparse(self, z):
        z = z.transpose(1,2)
        z = z.contiguous().view(-1, self.args.latent_dim)
        return z
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