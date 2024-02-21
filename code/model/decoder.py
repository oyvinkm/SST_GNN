import numpy as np
import torch
from torch import nn
from torch.nn import LayerNorm, Linear, ReLU, Sequential, LeakyReLU
from torch_geometric.data import Batch
from torch_geometric.nn.norm import BatchNorm
from loguru import logger

try:
    from utility import MessagePassingLayer, unpool_edge, Unpool
except:
    from .utility import MessagePassingLayer, unpool_edge, Unpool

class Decoder(nn.Module):
    def __init__(self, args, m_ids, m_gs, e_s):
        super(Decoder, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.max_hidden_dim = args.hidden_dim * 2 ** args.ae_layers
        # Pre computed node mask and edge_mask from bi-stride pooling
        self.m_ids, self.m_gs, self.e_s = m_ids, m_gs, e_s
        self.ae_layers = args.ae_layers
        self.n = args.n_nodes
        self.layers = nn.ModuleList()
        self.out_feature_dim = args.out_feature_dim
        self.latent_vec_dim = args.max_latent_nodes
        self.latent_edge_dim = args.max_latent_edges
        self.mpl_bottom = MessagePassingLayer(hidden_dim = args.latent_dim, 
                                              latent_dim=self.max_hidden_dim, 
                                              args=args)
        # self.linear_up_mlp_edge = Sequential(Linear(1, 500),
        #                             LeakyReLU(),
        #                             Linear(500, self.latent_edge_dim))
        self.edge_encoder = Sequential(Linear(args.in_dim_edge , self.hidden_dim),
                              ReLU(),
                              Linear(self.hidden_dim, self.latent_dim),
                              LayerNorm(self.latent_dim)
                              )
        self.linear_up_mlp = Sequential(Linear(1, 64),
                                    LeakyReLU(),
                                    Linear(64, self.latent_vec_dim))

        for i in range(self.ae_layers):
            if i == self.ae_layers - 1:
                up_nodes = self.n
            else:
                up_nodes = m_ids[args.ae_layers - i- 2]
            self.layers.append(Res_up(channel_in=self.max_hidden_dim // 2 ** i,
                                      channel_out=self.max_hidden_dim // 2 ** (i+1),
                                      args = args,
                                      m_id = m_ids[args.ae_layers - i - 1],
                                      m_g = m_gs[args.ae_layers - i - 1],
                                      e_idx = e_s[args.ae_layers -i - 1],
                                      up_nodes = up_nodes)
                                      )

        self.final_layer = MessagePassingLayer(hidden_dim = self.hidden_dim, latent_dim = self.hidden_dim, args=self.args)
        self.out_node_decoder = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            LeakyReLU(),
            Linear(self.hidden_dim // 2, self.out_feature_dim),
            LayerNorm(self.out_feature_dim),
        )
        self.out_edge_encoder = Sequential(
            Linear(self.hidden_dim, self.hidden_dim // 2),
            LeakyReLU(),
            Linear(self.hidden_dim // 2, 3),
            LayerNorm(3),
        )

    def from_latent_vec(self, z_x):
        x = self.linear_up_mlp(z_x)
        # e = self.linear_up_mlp_edge(z_x)
        x = self.batch_to_sparse(x)
        # e = self.batch_to_sparse(e)
        #return x, e
        return x
    def trim_nodes(self, b_data):
        b_lst = Batch.to_data_list(b_data)
        data_lst = []
        for idx, data in enumerate(b_lst):
            node_mask = self.m_ids[-1][data.trajectory]
            #edge_mask = self.m_gs[-1][data.trajectory].shape[-1]
            data.x = data.x[:len(node_mask)]
            data.weights = data.weights[:len(node_mask)]
            #data.edge_attr = data.edge_attr[:edge_mask]
            data_lst.append(data)
        return Batch.from_data_list(data_lst)
    
    def batch_to_sparse(self, z):
        z = z.transpose(1,2)
        z = z.contiguous().view(-1, self.args.latent_dim)
        return z

    def forward(self, b_data, z):
        # Set edge weights to 1
        b_data.weights = torch.ones_like(b_data.weights)
        if self.args.latent_space:
            #z_x, z_e = self.from_latent_vec(z)
            z_x = self.from_latent_vec(z)
            b_data.x = z_x
            #b_data.edge_attr = z_e
        b_data = self.trim_nodes(b_data)
        logger.debug(f'pre edge encoding : {b_data.edge_attr.shape}')
        b_data.edge_attr = self.edge_encoder(b_data.edge_attr)
        logger.debug(f'post edge encoding : {b_data.edge_attr.shape}')
        #logger.debug(f'b_data after edge encoding and node transform: {b_data}')
        logger.debug(f'{b_data=}')
        b_data = self.mpl_bottom(b_data)
        for i in range(self.ae_layers):
           b_data = self.layers[i](b_data)
        b_data = self.final_layer(b_data) #
        b_data.x = self.out_node_decoder(b_data.x) #
        b_data.edge_attr = self.out_edge_encoder(b_data.edge_attr)
        return b_data
    
    
class Res_up(nn.Module):

    def __init__(self, channel_in, channel_out, args, m_id, m_g, e_idx, up_nodes):
        super(Res_up, self).__init__()
        self.m_id = m_id
        self.m_g = m_g
        self.e_idx = e_idx
        self.args = args
        self.up_nodes = up_nodes
        self.mpl1 = MessagePassingLayer(channel_in, channel_out//2, args)
        self.mpl2 = MessagePassingLayer(channel_out//2, channel_out, args)
        self.mpl_skip = MessagePassingLayer(channel_in, channel_out, args)
        self.unpool = Unpool()
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()
        self.bn_nodes = BatchNorm(in_channels = channel_out)
        self.bn_edges = BatchNorm(in_channels = channel_out)

    
    def _bi_up_pool_batch(self, b_data):
        b_lst = b_data.to_data_list()
        batch_lst = []
        for idx, data in enumerate(b_lst):
            g, mask = self.m_g[data.trajectory], self.m_id[data.trajectory]
            up_nodes = self.up_nodes if isinstance(self.up_nodes, int) else len(self.up_nodes[data.trajectory])
            data.x = self.unpool(data.x, up_nodes, mask)
            data.weights = self.unpool(data.weights, up_nodes, mask)
            data.edge_index, data.edge_attr = unpool_edge(g, data.edge_attr, self.e_idx[data.trajectory], self.args)
            batch_lst.append(data)
        b_data = Batch.from_data_list(batch_lst).to(self.args.device)
        return b_data
    
    def forward(self, b_data):
        b_skip = self.mpl_skip(self._bi_up_pool_batch(b_data.clone()))
        b_data = self.mpl1(b_data)
        b_data = self._bi_up_pool_batch(b_data)
        b_data = self.mpl2(b_data)
        b_data.x = self.bn_nodes(b_data.x + b_skip.x)
        b_data.edge_attr = self.bn_edges(b_data.edge_attr + b_skip.edge_attr)
        b_data.x = self.act1(b_data.x)
        b_data.edge_attr = self.act2(b_data.edge_attr) 
        return b_data