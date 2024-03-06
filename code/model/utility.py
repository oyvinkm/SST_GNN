from functools import wraps
import types

from loguru import logger
import numpy as np
import scipy
import torch

from matplotlib import pyplot as plt

from torch import nn
from functools import wraps
from loguru import logger
from torch_geometric.data import Batch
from torch.nn import LayerNorm, Linear, ReLU, Sequential, LeakyReLU
from torch_geometric.nn.conv import GraphConv, MessagePassing, SAGEConv
from torch_geometric.nn.pool import ASAPooling, SAGPooling, TopKPooling
from torch_geometric.utils import degree, coalesce,to_dense_adj
from torch_scatter import scatter
from dataclasses import dataclass, astuple








def pool_edge(m_id, edge_index, edge_attr: torch.Tensor, aggr: str="mean"):
    r"""Pools the edges of a graph to a new set of edges using the idxHR_to_idxLR mapping.

    Args:
        idxHR_to_idxLR (torch.Tensor): A mapping from the old node (or higher resolution) indices to the new (or lower resolution) node indices.
        edge_index (torch.Tensor): The old edge indices.
        edge_attr (torch.Tensor): The old edge attributes.
        aggr (str, optional): The aggregation method. Can be "mean" or "sum". Defaults to "mean".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The new edge indices and attributes.
    """
    num_nodes = len(m_id)# number of nodes in the lower resolution graph
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, reduce=aggr) # aggregate edges
    return edge_index, edge_attr



def _adj_mat_to_flat_edge(adj_mat):
  if len(adj_mat.shape) == 2:
    s,r = np.where(adj_mat)
    return torch.tensor(np.array([s, r]), dtype=torch.int64)
  elif len(adj_mat.shape) == 3:
    s,r, p = np.where(adj_mat.astype(bool))
    return torch.tensor([s, r, p], dtype=torch.int64)


def adj_degree(g):
  # For efficiency
  g = scipy.sparse.coo_array(g)
  g.setdiag(1)
  # Compressed sparse row format
  g = g.tocsr().astype(float)

  # Dot product/matrix multiplication
  g = g@g
  g.setdiag(0)
  return g.toarray()

def unpool_edge(edge_index, edge_attr, e_idx, args):
  g = to_dense_adj(edge_index).detach().numpy().squeeze()
  # Matrix Magic
  g = adj_degree(g)
  # To Flat Edge
  g = _adj_mat_to_flat_edge(g)
  # Create empty array of all possible edges 
  new_edge_attr = torch.zeros((g.shape[1], edge_attr.shape[-1]), dtype=torch.float32).to(args.device)
  # Creates a 1d list of the 2d list
  c = g.T[:,0]+g.T[:,1]*1j
  # Creates a 1d list of the 2d list
  d = edge_index.T[:,0]+edge_index.T[:,1]*1j
  # Mask out actual edges
  res = np.in1d(c,d)
  mask = np.where(res)[0]
  # Fill out edge attributes of prev resolution
  new_edge_attr[e_idx,:] = edge_attr
  # Mask edge_attributes
  new_edge_attr = new_edge_attr[mask]
  new_edge_index = g[:, mask]
  return new_edge_index, new_edge_attr

class MessagePassingEdgeConv(MessagePassing):
    def __init__(self, channel_in, channel_out, args):
        super(MessagePassingEdgeConv, self).__init__()
        self.messagePassing = MessagePassingBlock(hidden_dim=channel_in, latent_dim = channel_out, num_blocks=args.num_blocks, args = args)
        self.edge_conv = WeightedEdgeConv()
        self.args = args
    
    def forward(self, b_data):
        x, g, w = b_data.x, b_data.edge_index, b_data.weights
        x = self.messagePassing(x, g)
        ew, w = self.edge_conv.cal_ew(w, g)
        # Does edge convolution on nodes with edge weigths
        x = self.edge_conv(x, g, ew)
        # Does edge convolution on position with edge weights
        if len(w.shape) < 2:
            w = w.unsqueeze(dim = 1)
        b_data.weights = w
        b_data.x = x
        return b_data
    

class GCNConv(MessagePassing):
    """
    Classic MessagePassing/Convolution
    """

    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr="add")  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, b_data):
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, E]
        x = b_data.x
        edge_index = b_data.edge_index

        # Step 1: Add self-loops to the adjacency matrix.
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [num_edges, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [num_nodes, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

class MessagePassingBlock(torch.nn.Module):
    """
    Just combines n number of message passing layers
    """

    def __init__(self, channel_in, channel_out, args, num_blocks=None):
        super(MessagePassingBlock, self).__init__()
        if num_blocks is None:
            self.num_blocks = args.num_blocks
        else:
            self.num_blocks = num_blocks
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.processor = nn.ModuleList()
        assert self.num_blocks >= 1, "Number of message passing layers is not >=1"
        
        processor_layer = self.build_processor_model()
        for i in range(self.num_blocks):
            if i == 0:
                self.processor.append(processor_layer(self.channel_in, self.channel_out))
            else:
                self.processor.append(processor_layer(self.channel_out, self.channel_out))

    def build_processor_model(self):
        return SAGEConv

    def forward(self, b_data):
        # Step 1: encode node/edge features into latent node/edge embeddings
        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_blocks):
            b_data.x = self.processor[i](b_data.x, b_data.edge_index)
        return b_data

class WeightedEdgeConv(MessagePassing):
    """
    Weighted Edge Convolution used for pooling and unpooling.
    """

    def __init__(self):
        super().__init__(aggr="add", flow="target_to_source")

    def forward(self, x, g, ew, aggragating=True):
        # aggregating: False means returning
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            weighted_info = x[:, i] if aggragating else x[:, j]
        elif len(x.shape) == 2:
            weighted_info = x[i] if aggragating else x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        weighted_info *= ew.unsqueeze(-1)
        target_index = j if aggragating else i
        aggr_out = scatter(
            weighted_info, target_index, dim=-2, dim_size=x.shape[-2], reduce="sum"
        )
        return aggr_out

    @torch.no_grad()
    def cal_ew(self, w, g):
        if w is None:
            w = torch.ones_like()
        deg = degree(g[0], dtype=torch.float, num_nodes=w.shape[0])
        normed_w = w.squeeze(-1) / deg
        i = g[0]
        j = g[1]
        w_to_send = normed_w[i]
        eps = 1e-12
        aggr_w = (
            scatter(w_to_send, j, dim=-1, dim_size=normed_w.size(0), reduce="sum") + eps
        )
        ec = w_to_send / aggr_w[j]
        return ec, aggr_w


    

class MessagePassingLayer(torch.nn.Module):
    """
    Kinda like a U-Net but with Message Passing Blocks.
    The Multiscale Autoencoder consists of multiple of these
    """

    def __init__(self, hidden_dim, latent_dim, args, bottom = False, first_up = False):
        super(MessagePassingLayer, self).__init__()
        self.l_n = args.mpl_layers
        self.args = args
        self.bottom = bottom
        self.first_up = first_up
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        """ if args.latent_dim is not None and (bottom or  first_up):
            if first_up:
                self.hidden_dim = args.latent_dim
                self.latent_dim = args.hidden_dim
            else:
                self.latent_dim = args.latent_dim
        else:
            self.latent_dim = args.hidden_dim """
        self.num_blocks = args.num_blocks
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.bottom_gmp = MessagePassingBlock(channel_in = self.latent_dim, channel_out = self.latent_dim, args=args)
        self.edge_conv = WeightedEdgeConv()
        self.pool = self._pooling_strategy()
        self.pools = nn.ModuleList()
        if self.args.mpl_ratio is None:
            self.mpl_ratio = 0.5
        else:
            self.mpl_ratio = self.args.mpl_ratio

        for i in range(self.l_n):
            if i == 0:
                self.down_gmps.append(
                        MessagePassingBlock(channel_in=self.hidden_dim, channel_out = self.latent_dim, args=args)
                    )
            else:
                self.down_gmps.append(
                       MessagePassingBlock(channel_in=self.latent_dim, channel_out = self.latent_dim, args=args)
                    )
            self.up_gmps.append(
                MessagePassingBlock(channel_in=self.latent_dim, channel_out=self.latent_dim, args=args)
            )
            self.unpools.append(Unpool())
            if self.args.pool_strat == "ASA":
                self.pools.append(
                    self.pool(
                        in_channels=self.latent_dim, ratio=self.mpl_ratio, GNN=GraphConv
                    )
                )
            else:
                self.pools.append(
                    self.pool(self.latent_dim, self.mpl_ratio)
                )

    def forward(self, b_data):
        """Forward pass through Message Passing Layer"""
        # Maybe make into a dict for readability? 
        down_outs = []
        cts = []
        down_masks = []
        down_gs = []
        batches = []
        ws = []
        attr = []
        b_data.weights = b_data.x.new_ones((b_data.x.shape[-2], 1)) if b_data.weights is None else b_data.weights
        for i in range(self.l_n):
            """ h = b_data.x
            g = b_data.edge_index """
            # This should return x as we're using GCNConv
            b_data = self.down_gmps[i](b_data)
            # record the infor before aggregation
            down_outs.append(b_data.x)
            down_gs.append(b_data.edge_index)
            batches.append(b_data.batch)
            ws.append(b_data.weights)
            attr.append(b_data.edge_attr)

            # aggregate then pooling
            # Calculates edge and node weigths
            if self.args.edge_conv:
                ew, w = self.edge_conv.cal_ew(b_data.weights, b_data.edge_index)
                b_data.weights = w
                # Does edge convolution on nodes with edge weigths
                b_data.x = self.edge_conv(b_data.x, b_data.edge_index, ew)
                # Does edge convolution on position with edge weights
                cts.append(ew)
            #b_data.x = h
            if self.args.pool_strat == "ASA":
                x, edge_index, edge_attr, batch, index = self.pools[i](
                    b_data.x, b_data.edge_index, b_data.edge_attr, b_data.batch
                )
                down_masks.append(index)
                b_data.x = x
                b_data.edge_index = edge_index
                b_data.edge_attr = edge_attr
                b_data.batch = batch
                b_data.weights = b_data.weights[index]
            else: 
                # Removed edge_attr with _
                x, edge_index, _, batch, index, _ = self.pools[i](
                    x = b_data.x, edge_index = b_data.edge_index, batch = b_data.batch
                )
                down_masks.append(index)
                b_data.x = x
                b_data.edge_index = edge_index
                # b_data.edge_attr = edge_attr
                b_data.batch = batch
                b_data.weights = b_data.weights[index]
        b_data = self.bottom_gmp(b_data)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            # Unpooling
            b_data.x = self.unpools[i](
                b_data.x, down_outs[up_idx].shape[0], down_masks[up_idx]
            )
            # Old Edge
            b_data.edge_index = down_gs[up_idx]
            #Edge Convolution
            if self.args.edge_conv:
                b_data.x = self.edge_conv(b_data.x, b_data.edge_index, cts[up_idx], aggragating=False)
            # Message Passing
            # Skip connection batch
            b_data.batch = batches[up_idx]
            # Skip connection weights
            b_data.weights = ws[up_idx]
            # b_data.edge_attr = attr[up_idx]
            b_data = self.up_gmps[i](b_data)
            b_data.x = b_data.x.add(down_outs[up_idx])
            #b_data.x = h
            #b_data.edge_index = tmp_g
        #b_data.edge_attr = edge_attr
        return b_data
    
    def _pooling_strategy(self):
        if self.args.pool_strat == "ASA":
            pool = ASAPooling
        elif self.args.pool_strat == "SAG":
            pool = SAGPooling
        else:
            pool = TopKPooling
        return pool

class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,  **kwargs):
        super(ProcessorLayer, self).__init__(  **kwargs )
        """
        in_channels: dim of node embeddings [128], out_channels: dim of edge embeddings [128]

        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Note that the node and edge encoders both have the same hidden dimension
        # size. This means that the input of the edge processor will always be
        # three times the specified hidden dimension
        # (input: adjacent node embeddings and self embeddings)
        self.edge_mlp = Sequential(Linear( 3* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))

        self.node_mlp = Sequential(Linear(in_channels + out_channels, out_channels),
                                   ReLU(),
                                   Linear(out_channels, out_channels),
                                   LayerNorm(out_channels))


        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, b_data, size = None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shape [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [edge_num, in_channels]

        """
        x = b_data.x
        edge_index = b_data.edge_index
        edge_attr = b_data.edge_attr
        arguments = {'dim_size' : (x.size(0), self.in_channels + self.out_channels)}
        out, updated_edges = self.propagate(edge_index = edge_index, 
                                            x = x, 
                                            edge_attr = edge_attr, 
                                            size = (x.size(0), x.size(0)), 
                                            **arguments) # out has the shape of [E, out_channels]


        updated_nodes = torch.cat([x,out],dim=1)        # Complete the aggregation through self-aggregation

        updated_nodes = self.node_mlp(updated_nodes)    # residual connection

        b_data.x = updated_nodes
        b_data.edge_attr = updated_edges
        return b_data

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        """

        updated_edges=torch.cat([x_i, x_j, edge_attr], dim = 1) # tmp_emb has the shape of [E, 3 * in_channels]
        updated_edges=self.edge_mlp(updated_edges)

        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size = None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """
        # The axis along which to index number of nodes.
        node_dim = 0

        out = scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'sum', dim_size=dim_size)
        return out, updated_edges


class LatentVecLayer(nn.Module):
    def __init__(self, hidden_dim, latent_dim, max_dim):
        super(LatentVecLayer, self).__init__()
        self.hidden_dim = hidden_dim 
        self.latent_dim = latent_dim
        self.max_dim = max_dim

        self.hidden_dim_mlp = Sequential(Linear(self.hidden_dim, self.hidden_dim // 2),
                              #ReLU(),
                              Linear(self.hidden_dim // 2, 1),
                              )
        self.latent_dim_mlp = Sequential(Linear(self.max_dim, self.max_dim // 2),
                              #ReLU(),
                              Linear(self.max_dim // 2, self.latent_dim),
                              )
        self.act = LeakyReLU()

    def forward(self, b_data):
        # Store b_data to transpoes each latent vec in batch
        
        b_data = b_data.clone()
        x = b_data.x
        logger.debug(f'working on nodes : {x.shape}')
        # Reduce hidden dimensions to 1 for each node 
        x = self.hidden_dim_mlp(x)
        logger.debug(f'after hidden_mlp {x.shape}')
        # Transpose 
        b_size = len(torch.unique(b_data.batch))
        print(b_size)
        print(self.max_dim)
        x = x.view(b_size, self.max_dim)
        # x = self.batch_to_dense_transpose(b_data)
        # Reduce to latent_dim
        x = self.latent_dim_mlp(x)
        logger.debug(f'After latent: {x.shape}')
        # Return latent vector
        #return self.act(x)
        return self.act(x).unsqueeze(dim = -1)

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





#################### Direction training ##########################
from enum import Enum
import json
import os



def vgae_with_shift(vgae_factory):
    """ 
    Wraps the vgae with the add_forward_with_shift
    It's also called a decorator function
    """
    @wraps(vgae_factory)
    def wrapper(*args, **kwargs):
        vgae = vgae_factory(*args, **kwargs)
        add_forward_with_shift(vgae)
        return vgae

    return wrapper

def add_forward_with_shift(generator):
    """
    Used in vgae_with_shift.
    Shifts a vector in the latent space.
    """
    def gen_shifted(self, b_data, z, shift, *args, **kwargs):
        return self.forward(b_data, z + shift, *args, **kwargs)

    # Creates these attributes for our generetor
    # the gen_shifted function is bound to the generator
    # It can be reread here: https://stackoverflow.com/questions/46525069/how-is-types-methodtype-used
    # Doesn't seem to important
    generator.dim_z = generator.latent_dim
    generator.gen_shifted = types.MethodType(gen_shifted, generator)
    generator.dim_shift = generator.latent_dim



class DeformatorType(Enum):
    FC = 1
    LINEAR = 2
    ID = 3
    ORTHO = 4
    PROJECTIVE = 5
    RANDOM = 6

DEFORMATOR_TYPE_DICT = {
    'fc': DeformatorType.FC,
    'linear': DeformatorType.LINEAR,
    'id': DeformatorType.ID,
    'ortho': DeformatorType.ORTHO,
    'proj': DeformatorType.PROJECTIVE,
    'random': DeformatorType.RANDOM,
}

def torch_expm(A):
    """Only used in the deformator in case the Deformator is of type ORTHO"""
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1, 2), keepdim=True))

    # Scaling step
    maxnorm = torch.tensor([5.371920351148152], dtype=A.dtype, device=A.device)
    zero = torch.tensor([0.0], dtype=A.dtype, device=A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    A_scaled = A / 2.0 ** n_squarings
    n_squarings = n_squarings.flatten().type(torch.int64)

    # Pade 13 approximation
    U, V = torch_pade13(A_scaled)
    P = U + V
    Q = -U + V
    R, _ = torch.solve(P, Q)

    # Unsquaring step
    res = [R]
    for i in range(int(n_squarings.max())):
        res.append(res[-1].matmul(res[-1]))
    R = torch.stack(res)
    expmA = R[n_squarings, torch.arange(n_A)]
    return expmA[0]

def make_noise(batch, dim, truncation = None):
    """Creates a random latent_vector of size equal to batch X dim"""
    if isinstance(dim, int):
        dim = [dim]
    if truncation is None or truncation == 1.0:
        return torch.randn([batch] + dim+[1,1])

class MeanTracker(object):
    """Tracks the mean of the value in the list : values."""
    def __init__(self, name):
        self.values = []
        self.name = name

    def add(self, val):
        self.values.append(float(val))

    def mean(self):
        return np.mean(self.values)

    def flush(self):
        mean = self.mean()
        self.values = []
        return self.name, mean

@torch.no_grad()
def interpolate(G, z, shifts_r, shifts_count, dim, deformator=None, with_central_border=False, device='cpu'):
    """Used by make_interpolation_chart"""
    shifted_images = []
    for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
        if deformator is not None:
            latent_shift = deformator(one_hot(deformator.input_dim, shift, dim).to(device))
        else:
            latent_shift = one_hot(G.dim_shift, shift, dim).to(device)
        shifted_image = G.gen_shifted(z, latent_shift).cpu()[0]
        if shift == 0.0 and with_central_border:
            shifted_image = add_border(shifted_image)

        shifted_images.append(shifted_image)
    return shifted_images

@torch.no_grad()
def make_interpolation_chart(G, deformator=None, z=None,
                             shifts_r=10.0, shifts_count=5,
                             dims=None, dims_count=10, texts=None, device='cpu', **kwargs):
    """Creates a figure that includes some interpolation"""
    with_deformation = deformator is not None
    if with_deformation:
        deformator_is_training = deformator.training
        deformator.eval()
    z = z if z is not None else make_noise(1, G.dim_z).to(device)

    if with_deformation:
        original_img = G(z).cpu()
    else:
        original_img = G(z).cpu()
    imgs = []
    if dims is None:
        dims = range(dims_count)
    for i in dims:
        imgs.append(interpolate(G, z, shifts_r, shifts_count, i, deformator, device=device))

    rows_count = len(imgs) + 1
    fig, axs = plt.subplots(rows_count, **kwargs)

    axs[0].axis('off')
    axs[0].imshow(to_image(original_img, True))

    if texts is None:
        texts = dims
    for ax, shifts_imgs, text in zip(axs[1:], imgs, texts):
        ax.axis('off')
        plt.subplots_adjust(left=0.5)
        ax.imshow(to_image(make_grid(shifts_imgs, nrow=(2 * shifts_count + 1), padding=1), True))
        ax.text(-20, 21, str(text), fontsize=10)

    if deformator is not None and deformator_is_training:
        deformator.train()

    return fig

def fig_to_image(fig):
    """creates an image from a figure"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)

class ShiftDistribution(Enum):
    NORMAL = 0,
    UNIFORM = 1,


@dataclass
class LatentVector:
    z : torch.TensorType
    t : list

    def __iter__(self):
        return iter(zip(self.z, self.t))
    def __repr__(self) -> str:
        return f'z : {self.z.shape},  t : {self.t}'
