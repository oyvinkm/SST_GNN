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
from model.utility import vgae_with_shift
try:
    from encoder import Encoder
    from decoder import Decoder
except:
    from .encoder import Encoder
    from .decoder import Decoder
from dataprocessing.dataset import MeshDataset
import os


class MultiScaleAutoEncoder(nn.Module):
    """
    Multiscale Auto Encoder consist of n_layer of Message Passing Layers (MPL) with
    pooling and unpooling operations in between in order to obtain a coarse latent
    representation of a graph. Uses an Multilayer Perceptron (MLP) to compute node and
    edge features.
    Encode: G_0 -> MLP -> MPL -> TopKPool ... MPL -> G_l -> Z_l
    Decode: G_l -> MPL -> Unpool .... -> MPL -> MLP -> G'_0 -> 
    """

    def __init__(self, args, m_ids, m_gs, e_s):
        super().__init__()
        self.args = args
        self.encoder = Encoder(args, m_ids, m_gs)
        self.decoder = Decoder(args, m_ids, m_gs, e_s)
        self.placeholder = Batch.from_data_list([Data(x=torch.ones(len(m_ids[-1]), args.latent_dim), 
                                                                   edge_index = m_gs[-1],
                                                                   weights = torch.ones(len(m_ids[-1])))])
        
    def forward(self, b_data, Train=True):

        kl, z, b_data = self.encoder(b_data, Train)
        b_data = self.decoder(b_data, z)
        return b_data, kl
    

# Not sure why I should wrap it in this function
@vgae_with_shift
def make_vgae(args, m_ids, m_gs, e_s):
    dec = Decoder(args, m_ids, m_gs, e_s)
    dec = dec.to(args.device)
    decoder_file = f"../logs/model_chkpoints/{args.model_file}"
    checkpoint = torch.load(decoder_file)
    dec.load_state_dict(checkpoint)
    dec.eval()
    return dec