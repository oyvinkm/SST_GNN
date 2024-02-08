import numpy as np
import torch
from torch import nn
from torch.nn import LayerNorm, Linear, ReLU, Sequential, LeakyReLU
from torch_geometric.data import Batch
from loguru import logger
import os
try:
    from utility import MessagePassingLayer, pool_edge
except:
    from .utility import MessagePassingLayer, pool_edge
    


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
        self.latent_vec_dim = len(m_ids[-1])
        self.b = args.batch_size
        self.layers = nn.ModuleList()

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
                                                latent_dim = self.latent_dim, 
                                                args=self.args, 
                                                bottom=True)
        self.mlp_logvar = Sequential(Linear(self.latent_vec_dim, 64),
                        LeakyReLU(),
                        Linear(64, 1))

        self.mlp_mu = Sequential(Linear(self.latent_vec_dim, 64),
                        LeakyReLU(),
                        Linear(64, 1))

    def forward(self, b_data, Train = True):
        b_data.x = self.node_encoder(b_data.x)
        b_data.edge_attr = self.edge_encoder(b_data.edge_attr)

        for i in range(self.ae_layers): 
            b_data = self.layers[i](b_data)
        b_data = self.bottom_layer(b_data) #

        if Train:
            x_t = self.batch_to_dense_transpose(b_data)
            mu = self.mlp_mu(x_t)
            log_var = self.mlp_logvar(x_t)
            z = self.sample(mu, log_var)
            kl = torch.mean(-0.5 * torch.sum(1+log_var-mu**2-log_var.exp(), dim=1), dim=0)

        else:
            x_t = self.batch_to_dense_transpose(b_data)
            z = self.mlp_mu(x_t)
            kl = None

        # self.save_bdata(b_data)

        return kl, z, b_data



    def save_bdata(self, b_data):
        PATH = os.path.join(self.args.graph_structure_dir, 'b_data.pt')
        if not os.path.isfile(PATH):
            torch.save(b_data, PATH)

    def sample(self, mu, logvar):
        """Shamelessly stolen from https://github.com/julschoen/Latent-Space-Exploration-CT/blob/main/Models/VAE.py"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def batch_to_dense_transpose(self, b_data):
        count = np.unique(b_data.batch.cpu(), return_counts= True)
        count = list(zip(count[0], count[1]))
        b_lst = []
        for b, len in count:
            start = b*len
            end = (b+1)*len 
            b_lst.append(b_data.x[start:end].T)
        batch = torch.stack(b_lst)
        return batch

class Res_down(nn.Module):
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

    def _bi_pool_batch(self, b_data):
        b_lst = Batch.to_data_list(b_data)
        data_lst = []
        g = self.m_g
        if not torch.is_tensor(self.m_g):
            g = torch.tensor(self.m_g)
        mask = self.m_id
        for idx, data in enumerate(b_lst):
            data.x = data.x[mask]
            data.weights = data.weights[mask]
            data.edge_index, data.edge_attr = pool_edge(mask, g, data.edge_attr)
            data_lst.append(data)
        return Batch.from_data_list(data_lst).to(self.args.device) 
    
    def forward(self, b_data):
        b_skip = self._bi_pool_batch(b_data.clone())
        b_skip = self.mpl_skip(b_skip) # out = channel_out
        b_data = self.mpl1(b_data)
        b_data = self._bi_pool_batch(b_data)
        b_data = self.mpl2(b_data)
        b_data.x = self.act1(b_data.x + b_skip.x)
        return b_data