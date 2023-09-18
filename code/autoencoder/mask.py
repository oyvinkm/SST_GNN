import torch
from typing import List, Optional, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import dropout_edge, dropout_node
import numpy as np
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

"""TODO:
    - Implement and understand the loss functions for Edge and NodeMasking
"""

@functional_transform('edge_mask')
class EdgeMask(BaseTransform):
    r"""Removes each edge in the graph with the probability given in :obj:'p' 
        (functional name: :obj:`edge_mask`).

    Args:
        p (float): The probability to remove each edge
            (default: :obj:`0.5`)
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, dataobject : Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        edge_index, edge_mask = dropout_edge(dataobject.edge_index, self.p, force_undirected=True)
        print("edge_mask = ", edge_mask)
        print("edge_index =", edge_index)
        self.masked_edges = dataobject.edge_index[np.where(edge_mask == 0)] # save the edges where the edge_mask is False
        dataobject.edge_index = edge_index
        return dataobject

@functional_transform('node_mask')
class NodeMask(BaseTransform):
    r"""Removes each Node in the graph with the probability given in :obj:'p' 
        (functional name: :obj:`node_mask`).

    Args:
        p (float): The probability to remove each node
            (default: :obj:`0.5`)
    """
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, dataobject : Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        dataobject.edge_index, _, node_mask = dropout_node(dataobject.edge_index, p=self.p)
        x = dataobject.x[np.where(node_mask==1)]
        dataobject.x = x
        return dataobject

edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                           [1, 0, 2, 1, 3, 2]])

x = torch.tensor([[-1],[0],[1],[2]])
dat = Data(x, edge_index)

transform = EdgeMask(p=0.5)
data = transform(dat)

# print(transform.masked_edges)

print(data)

"""From 
https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

Note that edge_index, i.e. the tensor defining the source and target nodes of all edges, 
is not a list of index tuples. If you want to write your indices this way, you should 
transpose and call contiguous on it before passing them to the data constructor:"""