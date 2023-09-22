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

@functional_transform('AttributeMask')
class AttributeMask(BaseTransform):
    """
    Sets attributes of random nodes to 0. The nodes are chosen with probability p
    NB!: The attributes are additionally converted to 32 bit floats
    returns: obj: 'dataobject': the dataobject with masked_features
             obj: 'cached_features': the features that was given as input,
                   converted to 32 bit float
    """
    def __init__(self, p=0.1):
        self.p = p
    
    def __call__(self, dataobject : Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        dataobject.x = dataobject.x.to(torch.float32)
        cached_features = dataobject.x
        idx_train = torch.arange(dataobject.x.shape[0])
        nfeat = dataobject.x.shape[1]
        masked_nodes = torch.rand(idx_train.size(0)) <= self.p
        masked_indicator = torch.zeros(nfeat)
        dataobject.x[masked_nodes] = masked_indicator

        return dataobject, cached_features

@functional_transform('my_edge_mask')
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
        dataobject.edge_index = edge_index
        return dataobject




"""From 
https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

Note that edge_index, i.e. the tensor defining the source and target nodes of all edges, 
is not a list of index tuples. If you want to write your indices this way, you should 
transpose and call contiguous on it before passing them to the data constructor:"""