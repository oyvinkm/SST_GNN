from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import OptTensor
from torch_geometric.utils import dropout_edge


def filter_adj(
    row: Tensor, col: Tensor, edge_attr: OptTensor, mask: Tensor
) -> Tuple[Tensor, Tensor, OptTensor]:
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

# Edgemasking, better than the "Edge_Mask" defined further down
def dropout_adj(
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes: Optional[int] = None,
    training: bool = True,
) -> Tuple[Tensor, OptTensor]:
    r"""Stolen shamelessly from pytorch-geometric. We don't want to use the
    function from their library as it will be deprecated soon."""
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1 " f"(got {p}")

    if not training or p == 0.0:
        return edge_index, edge_attr

    row, col = edge_index

    mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        mask[row > col] = False

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0
        )
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


@functional_transform("AttributeMask")
class AttributeMask(BaseTransform):
    """
    Sets attributes of random nodes to 0. The nodes are chosen with probability p
    NB!: The attributes are additionally converted to 32 bit floats
    returns: obj: 'dataobject': the dataobject with masked_features
             obj: 'cached_features': the features that was given as input,
                   converted to 32 bit float
    """

    def __init__(self, p=0.1, device="cpu"):
        self.device = device
        self.p = p

    def __call__(self, dataobject: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        dataobject.x = dataobject.x.to(torch.float32)
        idx_train = torch.arange(dataobject.x.shape[0])
        nfeat = dataobject.x.shape[1]
        masked_nodes = torch.rand(idx_train.size(0)) <= self.p
        masked_indicator = torch.zeros(nfeat).to(self.device)
        dataobject.x[masked_nodes] = masked_indicator

        return dataobject

@functional_transform("FlipGraph")
class FlipGraph(BaseTransform):
    """Flips a graph horizontally. This changes the following attributes:
        - Velocities at nodes : x
        - Mesh positions for plotting : mesh_pos
        - Edge attributes between nodes : edge_attr.
    """

    def __call__(self, data: Data) -> Data:
        x = data.x.clone()
        mesh_pos = data.mesh_pos.clone()
        edge_attr = data.edge_attr.clone()
        x[..., 0] = -x[..., 0]
        edge_attr[...,0] *= -1
        mesh_pos[..., 0] = -mesh_pos[..., 0]

        data.edge_attr = edge_attr
        data.mesh_pos = mesh_pos
        data.x = x

        return data

@functional_transform("my_edge_mask")
class EdgeMask(BaseTransform):
    r"""Removes each edge in the graph with the probability given in :obj:'p'
        (functional name: :obj:`edge_mask`).

    Args:
        p (float): The probability to remove each edge
            (default: :obj:`0.5`)
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, dataobject: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        if dataobject.edge_attr is None:
            edge_index, _ = dropout_edge(
                dataobject.edge_index, self.p, force_undirected=True
            )
            dataobject.edge_index = edge_index
        else:
            edge_index, edge_attr = dropout_adj(
                dataobject.edge_index,
                dataobject.edge_attr,
                self.p,
                force_undirected=True,
            )
            dataobject.edge_index = edge_index
            dataobject.edge_attr = edge_attr
        return dataobject


"""From 
https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

Note that edge_index, i.e. the tensor defining the source and target nodes of all edges, 
is not a list of index tuples. If you want to write your indices this way, you should 
transpose and call contiguous on it before passing them to the data constructor:"""
