"""
    This file contains the classes used to build our Multi Scale Auto Encoder GNN.
"""
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
from encoder import Encoder
from decoder import Decoder
from encoder_alt import Encoder as EC
from decoder_alt import Decoder as DC


class MultiScaleAutoEncoder(nn.Module):
    """
    Multiscale Auto Encoder consist of n_layer of Message Passing Layers (MPL) with
    pooling and unpooling operations in between in order to obtain a coarse latent
    representation of a graph. Uses an Multilayer Perceptron (MLP) to compute node and
    edge features.
    Encode: G_0 -> MLP -> MPL -> TopKPool ... MPL -> G_l -> Z_l
    Decode: G_l -> MPL -> Unpool .... -> MPL -> MLP -> G'_0 -> 
    """

    def __init__(self, args, m_ids, m_gs):
        super().__init__()
        self.encoder = Encoder(args, m_ids, m_gs)
        self.decoder = Decoder(args, m_ids, m_gs)
        self.placeholder = Batch.from_data_list([Data(x=torch.ones(len(m_ids[-1]), args.latent_dim), 
                                                                   edge_index = torch.tensor(m_gs[-1]),
                                                                   weights = torch.ones(len(m_ids[-1])))])
        
    def forward(self, b_data, Train=True):
        kl, z, b_data = self.encoder(b_data, Train)
        b_data = self.decoder(b_data, z)
        return b_data, kl
    
# @gmsvae_with_shift
# def make_gmsvae(model_path, args, m_ids, g_ids):
#     model = MultiScaleAutoEncoder(args, m_ids, m_gs)
#     model.load_state_dict(torch.load(model_path))
#     model = model.eval()
#     return model

class MultiScaleAutoEncoder_Alt(nn.Module):
    """
    Multiscale Auto Encoder consist of n_layer of Message Passing Layers (MPL) with
    pooling and unpooling operations in between in order to obtain a coarse latent
    representation of a graph. Uses an Multilayer Perceptron (MLP) to compute node and
    edge features.
    Encode: G_0 -> MLP -> MPL -> TopKPool ... MPL -> G_l -> Z_l
    Decode: G_l -> MPL -> Unpool .... -> MPL -> MLP -> G'_0 -> 
    """

    def __init__(self, args, m_ids, m_gs):
        super().__init__()
        self.encoder = EC(args, m_ids, m_gs)
        self.decoder = DC(args, m_ids, m_gs)
        self.placeholder = Batch.from_data_list([Data(x=torch.ones(len(m_ids[-1]), args.latent_dim), 
                                                                   edge_index = torch.tensor(m_gs[-1]),
                                                                   weights = torch.ones(len(m_ids[-1])))])
        
    def forward(self, b_data, Train=True):
        kl, z, b_data = self.encoder(b_data, Train)
        b_data = self.decoder(b_data, z)
        return b_data, kl