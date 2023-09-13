import torch
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, dropout_node
import numpy as np



def EdgeMask(dataobject, p=0.5):
    # Example usage:
    """edge_index = torch.tensor([[0, 1],
                            [1, 0],
                            [1, 2],
                            [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())
    print(data.edge_index)

    EdgeMask(data, p=0.5)
    print("Edges that are left: ", data.edge_index)"""

    dataobject.edge_index, _= dropout_edge(dataobject.edge_index, p=p, force_undirected=True)
    return dataobject

def NodeMask(dataobject, p=0.5):
    # Example usage:
    """edge_index = torch.tensor([[0, 1],
                            [1, 0],
                            [1, 2],
                            [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())
    print(data.edge_index)

    NodeMask(data, p=0.5)
    print("Nodes that are left: ", data.x)"""

    dataobject.edge_index, _, node_mask = dropout_node(dataobject.edge_index, p=p)
    x = dataobject.x[np.where(node_mask==1)]
    dataobject.x = x
    return dataobject


"""From 
https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

Note that edge_index, i.e. the tensor defining the source and target nodes of all edges, 
is not a list of index tuples. If you want to write your indices this way, you should 
transpose and call contiguous on it before passing them to the data constructor:"""