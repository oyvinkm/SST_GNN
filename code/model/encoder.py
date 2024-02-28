import numpy as np
import torch

import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm, Linear, ReLU, Sequential, LeakyReLU
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.data import Batch
from loguru import logger
import os
try:
    from utility import MessagePassingLayer, pool_edge, Unpool, LatentVecLayer, LatentVector
except:
    from .utility import MessagePassingLayer, pool_edge, Unpool, LatentVecLayer, LatentVector
    


class Encoder(nn.Module):
    def __init__(self, args, m_ids, m_gs):
        super(Encoder, self).__init__()
        self.args = args
        self.m_ids, self.m_gs = m_ids, m_gs
        self.ae_layers = args.ae_layers
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.dim_z = self.latent_dim
        self.in_dim_node = args.in_dim_node
        self.in_dim_edge = args.in_dim_edge
        self.latent_node_dim = args.max_latent_nodes
        self.latent_edge_dim = args.max_latent_edges
        self.b = args.batch_size
        self.layers = nn.ModuleList()
        self.pad = Unpool()

        self.node_encoder = Sequential(
            Linear(self.in_dim_node, self.hidden_dim),
            LeakyReLU(),
            Linear(self.hidden_dim, self.hidden_dim),
            LayerNorm(self.hidden_dim),
        )
        self.edge_encoder = Sequential(Linear(self.in_dim_edge , self.hidden_dim),
                              ReLU(),
                              Linear( self.hidden_dim, self.hidden_dim),
                              LayerNorm(self.hidden_dim)
                              )
        for i in range(self.ae_layers):
            
            self.layers.append(Res_down(
                                        channel_in = self.hidden_dim * 2**i, 
                                        channel_out = self.hidden_dim * 2**(i + 1),
                                        m_id = self.m_ids[i],
                                        m_g = self.m_gs[i+1],
                                        args = args))
        self.bottom_layer = MessagePassingLayer(hidden_dim = self.hidden_dim * 2 ** self.ae_layers, 
                                                latent_dim = self.hidden_dim * 2 ** self.ae_layers, 
                                                args=self.args, 
                                                bottom=True)
        
        self.node_latent_mlp = LatentVecLayer(hidden_dim=self.hidden_dim * 2 ** self.ae_layers,
                                              latent_dim = self.latent_dim, max_dim = self.latent_node_dim, type='node')
        # self.edge_latent_mlp = LatentVecLayer(hidden_dim=self.hidden_dim * 2 ** self.ae_layers,
        #                                       latent_dim = self.latent_dim, max_dim = self.latent_edge_dim, type='edge')
        self.mlp_mu_nodes = Sequential(Linear(self.latent_dim, self.latent_dim),
                              ReLU(),
                              LayerNorm(self.latent_dim)
                              )
        self.mlp_logvar_nodes = Sequential(Linear(self.latent_dim, self.latent_dim),
                              ReLU(),
                              LayerNorm(self.latent_dim)
                              )
        # self.mlp_zip_node = Sequential(Linear(self.latent_vec_dim, 64),
        #                 LeakyReLU(),
        #                 Linear(64, self.zip_dim))

        #self.mlp_mu = Sequential(Linear(self.zip_dim*2, 1))
        # self.mlp_mu_nodes = Sequential(Linear(self.latent_node_dim, 64),
        #                       ReLU(),
        #                       Linear(64, 1),
        #                       LayerNorm(1)
        #                       )
        # self.mlp_logvar_nodes = Sequential(Linear(self.latent_node_dim, 64),
        #                       ReLU(),
        #                       Linear(64, 1),
        #                       LayerNorm(1)
        #                       )
        # self.mlp_mu_edges = Sequential(Linear(self.latent_edge_dim, self.latent_edge_dim // 2),
        #                       ReLU(),
        #                       Linear(self.latent_edge_dim//2, 64),
        #                       ReLU(),
        #                       Linear(64, 1),
        #                       LayerNorm(1)
        #                       ) 
        # self.mlp_logvar_edges = Sequential(Linear(self.latent_edge_dim, self.latent_edge_dim // 2),
        #                       ReLU(),
        #                       Linear(self.latent_edge_dim//2, 64),
        #                       ReLU(),
        #                       Linear(64, 1),
        #                       LayerNorm(1)
        #                       ) 

    def forward(self, b_data, Train = True):
        b_data.x = self.node_encoder(b_data.x)
        b_data.edge_attr = self.edge_encoder(b_data.edge_attr)

        for i in range(self.ae_layers): 
            b_data = self.layers[i](b_data)
        b_data = self.bottom_layer(b_data) #
        b_data = self.pad_nodes_edges(b_data)
        if Train:
            x_t = self.node_latent_mlp(b_data).transpose(1,2)
            logger.debug(f'Latent nodes : {x_t.shape}')
            # e_t = self.edge_latent_mlp(b_data)
            # logger.debug(f'Latent edges : {e_t.shape}')
            # Sampling latent vector for nodes and calculating KL-divergence
            mu_nodes = self.mlp_mu_nodes(x_t)
            log_var_nodes = self.mlp_logvar_nodes(x_t)
            z_nodes = self.sample(mu_nodes, log_var_nodes)
            kl = torch.mean(-0.5 * torch.sum(1+log_var_nodes-mu_nodes**2-log_var_nodes.exp(), dim=0), dim=1)
            # x_t, e_t = self.batch_to_dense_transpose(b_data)
            # mu_nodes = self.mlp_mu_nodes(x_t)
            # log_var_nodes = self.mlp_logvar_nodes(x_t)
            # logger.debug(f'Transposed : {x_t.shape}')
            # z_nodes = self.sample(mu_nodes, log_var_nodes)
            # kl_nodes = torch.mean(-0.5 * torch.sum(1+log_var_nodes-mu_nodes**2-log_var_nodes.exp(), dim=1), dim=0)
            # mu_edges = self.mlp_mu_edges(e_t)
            # log_var_edges = self.mlp_logvar_edges(e_t)
            # z_edges = self.sample(mu_edges, log_var_edges)
            # kl_edges = torch.mean(-0.5 * torch.sum(1+log_var_edges-mu_edges**2-log_var_edges.exp(), dim=1), dim=0)
            # logger.debug(f'{z_nodes.shape}, {z_edges.shape}')
            logger.debug(f'z before returning from encoder: {type(z_nodes)=} {z_nodes.shape}')
            z = LatentVector(z_nodes, b_data.trajectory)
            return kl, z , b_data

        else:
            x_t = self.node_latent_mlp(b_data).transpose(1,2)
            z_nodes = self.mlp_mu_nodes(x_t)
            kl = None
            return kl, z_nodes, b_data

        #self.save_bdata(b_data)

    def save_bdata(self, b_data):
        PATH = os.path.join(self.args.graph_structure_dir, 'b_data.pt')
        if not os.path.isfile(PATH):
            torch.save(b_data, PATH)

    def sample(self, mu, logvar):
        """Shamelessly stolen from https://github.com/julschoen/Latent-Space-Exploration-CT/blob/main/Models/VAE.py"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def pad_nodes_edges(self, b_data):
        b_lst = Batch.to_data_list(b_data)
        data_lst = []
        for idx, data in enumerate(b_lst):
            data.x = self.pad(data.x, self.latent_node_dim, np.arange(0, data.x.shape[0]))
            data.weights = self.pad(data.weights, self.latent_node_dim, np.arange(0, data.weights.shape[0]))
            # data.edge_attr = self.pad(data.edge_attr, self.latent_edge_dim, np.arange(0, data.edge_attr.shape[0]))
            data_lst.append(data)
        return Batch.from_data_list(data_lst).to(self.args.device)
    
    @torch.no_grad()
    def batch_to_dense_transpose(self, b_data):
        data_lst = Batch.to_data_list(b_data)
        b_node_lst = []
        # b_edge_lst = []
        for b in data_lst:
            node_vec = b.x.T
            edge_vec = b.edge_attr.T
            b_node_lst.append(node_vec)
            # b_edge_lst.append(edge_vec)
        node_batch = torch.stack(b_node_lst)
        # edge_batch = torch.stack(b_edge_lst)
        return node_batch

class Res_down(nn.Module):
    """Take m_id and m_g in forwad not"""
    def __init__(self, channel_in, channel_out, args, m_id, m_g):
        super(Res_down, self).__init__()
        self.m_id = m_id
        self.m_g = m_g
        self.args = args
        self.mpl1 = MessagePassingLayer(channel_in, channel_out // 2, args)
        self.mpl2 = MessagePassingLayer(channel_out // 2, channel_out, args)
        self.act1 = nn.ReLU()
        self.mpl_skip = MessagePassingLayer(channel_in, channel_out, args) # skip
        self.act2 = nn.ReLU()
        self.bn_nodes = BatchNorm(in_channels = channel_out)
        # self.bn_edges = BatchNorm(in_channels = channel_out)

    def _bi_pool_batch(self, b_data):
        b_lst = Batch.to_data_list(b_data)
        data_lst = []
        
        for idx, data in enumerate(b_lst):
            g = self.m_g[data.trajectory]
            mask = self.m_id[data.trajectory]
            data.x = data.x[mask]
            data.weights = data.weights[mask]
            data.edge_index, _ = pool_edge(mask, g, data.edge_attr)
            data_lst.append(data)
        return Batch.from_data_list(data_lst).to(self.args.device) 
    
    def forward(self, b_data):
        # Removed edge_attr
        b_skip = self._bi_pool_batch(b_data.clone())
        b_skip = self.mpl_skip(b_skip) # out = channel_out
        b_data = self.mpl1(b_data)
        b_data = self._bi_pool_batch(b_data)
        b_data = self.mpl2(b_data)
        b_data.x = self.bn_nodes(b_data.x + b_skip.x)
        # b_data.edge_attr = self.bn_edges(b_data.edge_attr + b_skip.edge_attr)
        b_data.x = self.act1(b_data.x)
        # b_data.edge_attr = self.act2(b_data.edge_attr)
        return b_data