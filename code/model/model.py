"""
    This file contains the classes used to build our Multi Scale Auto Encoder GNN.
"""

import torch
from model.decoder import Decoder
from model.encoder import Encoder
from torch import nn
from torch_geometric.data import Batch, Data


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
        self.placeholder = Batch.from_data_list(
            [
                Data(
                    x=torch.ones(len(m_ids[-1]), args.latent_dim),
                    edge_index=m_gs[-1],
                    weights=torch.ones(len(m_ids[-1])),
                )
            ]
        )

    def forward(self, b_data, Train=True):
        kl, z = self.encoder(b_data, Train)
        b_data = self.decoder(z)
        return b_data, kl
