import os

import numpy as np
import torch
from loguru import logger
from model.utility import (
    LatentVecLayer,
    LatentVector,
    MessagePassingLayer,
    Unpool,
)
from torch import nn
from torch.nn import SELU, LayerNorm, Linear, Sequential
from torch_geometric.data import Batch
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import SAGPooling


class Encoder(nn.Module):
    def __init__(self, args, m_ids, m_gs):
        """
        Initialize the Encoder class.

        Args:
            args: The arguments for the Encoder.
            m_ids: The module IDs.
            m_gs: The module GS.
        """
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
            SELU(),
            Linear(self.hidden_dim, self.hidden_dim),
        )
        for i in range(self.ae_layers):
            ratio = 0.5
            if i == self.ae_layers - 1:
                ratio = self.latent_node_dim
            self.layers.append(
                Res_down(
                    channel_in=self.hidden_dim * 2**i,
                    channel_out=self.hidden_dim * 2 ** (i + 1),
                    m_id=self.m_ids[i],
                    m_g=self.m_gs[i + 1],
                    args=args,
                    ratio=ratio,
                )
            )
        self.bottom_layer = MessagePassingLayer(
            hidden_dim=self.hidden_dim * 2**self.ae_layers,
            latent_dim=self.hidden_dim * 2**self.ae_layers,
            args=self.args,
            bottom=True,
        )

        self.node_latent_mlp = LatentVecLayer(
            hidden_dim=self.hidden_dim * 2**self.ae_layers,
            latent_dim=self.latent_dim,
            max_dim=self.latent_node_dim,
        )
        self.mlp_mu_nodes = Linear(self.latent_dim, self.latent_dim)
        self.mlp_logvar_nodes = Sequential(
            Linear(self.latent_dim, self.latent_dim), SELU(), LayerNorm(self.latent_dim)
        )
        self.mlp_logvar_nodes = Linear(self.latent_dim, self.latent_dim)

    def forward(self, b_data, Train=True):
        # Encode node embeddings
        b_data.x = self.node_encoder(b_data.x)

        # Message passing and pooling
        for i in range(self.ae_layers):
            b_data = self.layers[i](b_data)

        # Bottom layer message passing
        b_data = self.bottom_layer(b_data)  
        if Train:
            # (B, |V_L|, H) -> (B, 1, Latent dim)
            x_t = self.node_latent_mlp(b_data).transpose(1, 2)

            # Sampling latent vector for nodes and calculating KL-divergence
            mu_nodes = self.mlp_mu_nodes(x_t)
            log_var_nodes = self.mlp_logvar_nodes(x_t)
            z_nodes = self.sample(mu_nodes, log_var_nodes)
            kl = torch.mean(
                -0.5
                * torch.sum(
                    1 + log_var_nodes - mu_nodes**2 - log_var_nodes.exp(), dim=0
                ),
                dim=1,
            )
            z = LatentVector(z_nodes, b_data.trajectory)
            return kl, z, b_data

        else:
            x_t = self.node_latent_mlp(b_data).transpose(1, 2)
            z_nodes = self.mlp_mu_nodes(x_t)
            kl = None
            z = LatentVector(z_nodes, b_data.trajectory)
            return kl, z, b_data

    def sample(self, mu, logvar):
        """Shamelessly stolen from
        https://github.com/julschoen/Latent-Space-Exploration-CT/blob/main/Models/VAE.py
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Res_down(nn.Module):
    """
    The Res_down class is a part of a larger model and is 
    responsible for downsampling the input data. It uses message passing layers 
    (MessagePassingLayer) to propagate information through the graph structure. 
    The channel_in parameter specifies the number of input channels, 
    while the channel_out parameter specifies the number of output channels.
    The model incorporates residual connections between the downsampled layers.
    The class combines message passing, residual connections, activation functions, 
    batch normalization, and graph pooling to downsample the input data 
    while preserving important information and gradients.
    Args:
        channel_in (int): Number of input channels.
        channel_out (int): Number of output channels.
        args: Additional arguments.
        m_id: Identifier for the model.
        m_g: Graph for the model.
        ratio (float, optional): Ratio for pooling. Defaults to 0.5.
    """

    def __init__(self, channel_in, channel_out, args, m_id, m_g, ratio=0.5):
        super(Res_down, self).__init__()
        self.m_id = m_id
        self.m_g = m_g
        self.args = args
        self.mpl1 = MessagePassingLayer(channel_in, channel_out // 2, args)
        self.mpl2 = MessagePassingLayer(channel_out // 2, channel_out, args)
        self.act1 = SELU()
        self.mpl_skip = MessagePassingLayer(channel_in, channel_out, args)  # skip
        self.act2 = SELU()
        self.bn_nodes = BatchNorm(in_channels=channel_out)
        self.pool_skip = SAGPooling(in_channels=channel_in, ratio=ratio)
        self.pool = SAGPooling(in_channels=channel_out // 2, ratio=ratio)

    def _learnable_pool(self, b_data, skip=False):
        b_lst = Batch.to_data_list(b_data)
        data_lst = []
        for idx, data in enumerate(b_lst):
            if skip:
                x, edge_index, _, _, perm, _ = self.pool_skip(
                    x=data.x, edge_index=data.edge_index
                )
            else:
                x, edge_index, _, _, perm, _ = self.pool(
                    x=data.x, edge_index=data.edge_index
                )
            data.x = x
            data.edge_index = edge_index
            data.mesh_pos = data.mesh_pos[perm]
            data.weights = data.weights[perm]
            data_lst.append(data)
        return Batch.from_data_list(data_lst).to(self.args.device)

    def _bi_pool_batch(self, b_data):
        b_lst = Batch.to_data_list(b_data)
        data_lst = []

        for idx, data in enumerate(b_lst):
            g = self.m_g[data.trajectory]
            mask = self.m_id[data.trajectory]
            data.x = data.x[mask]
            data.mesh_pos = data.mesh_pos[mask]
            data.weights = data.weights[mask]
            data.edge_index = g
            data_lst.append(data)
        return Batch.from_data_list(data_lst).to(self.args.device)

    def forward(self, b_data):
        # Removed edge_attr
        if torch.any(torch.isnan(b_data.x)):
            logger.error("something is nan in start of Res_down")
        # NOTE: Implemented learnable pooling
        b_skip = self._learnable_pool(b_data.clone(), skip=True)
        b_skip = self.mpl_skip(b_skip)  # out = channel_out
        b_data = self.mpl1(b_data)
        b_data = self._learnable_pool(b_data)
        b_data = self.mpl2(b_data)
        b_data.x = b_data.x + b_skip.x
        if self.args.batch_norm:
            b_data.x = self.bn_nodes(b_data.x)
        b_data.x = self.act1(b_data.x)
        if torch.any(torch.isnan(b_data.x)):
            logger.error("something is nan at the end of Res_down")
        return b_data
