import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

"""https://github.com/ChandlerBang/SelfTask-GNN/blob/master/src/selfsl.py"""

# A random star graph
G = nx.star_graph(2)
G.add_edge(1, 3)
G.add_edge(2, 3)

A = nx.adjacency_matrix(G).todense()

# Is nnz just equal to A?
# It might actually be the number of nonzero entries in the adjacency matrix
nnz = np.sum(A)

# random array of numbers 1 to nnz
perm = np.random.permutation(nnz)

# The chance of removing an edge
mask_ratio = 0.1

# Calculates the amount of non-zero edges to preserve.
preserve_nnz = int(nnz*(1 - mask_ratio))

# Selecting which edges to mask
masked = perm[preserve_nnz: ]

# Den her forstår jeg ikke helt endnu, noget med at self.masked_edges nu indeholder vores masked edges, duh.
# self.masked_edges = (self.adj.row[masked], self.adj.col[masked])

# Perm2 contains the preserved edges
perm = perm[:preserve_nnz]

import scipy.sparse as sp

print(len(A), len(perm))

r_adj = sp.coo_matrix((perm, (perm, perm)), shape=A.shape)
print(A)
print(perm)
print(masked)
# print(masked)





# nx.draw_networkx(G)
# plt.show()