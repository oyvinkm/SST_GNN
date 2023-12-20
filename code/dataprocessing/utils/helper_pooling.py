import numpy as np
import torch
import scipy
import pickle
import os

_INF = _INF = 1 + 1e10
def _BFS_dist(adj_list, n_nodes, seed, mask=None):
    # mask: meaning only search within the subset indicated by it, any outside nodes are not reachable
    #       can be achieved by marking outside nodes as visited, dist to inf
    res = np.ones(n_nodes) * _INF
    vistied = [False for _ in range(n_nodes)]
    if isinstance(seed, list):
        for s in seed:
            res[s] = 0
            vistied[s] = True
        frontier = seed
    else:
        res[seed] = 0
        vistied[seed] = True
        frontier = [seed]
    
    depth = 0
    track = [frontier]
    while frontier:
        this_level = frontier
        depth += 1
        frontier = []
        while this_level:
            f = this_level.pop(0)
            for n in adj_list[f]:
                if not vistied[n]:
                    vistied[n] = True
                    frontier.append(n)
                    res[n] = depth
        # record each level
        track.append(frontier)

    return res, track


def _BFS_dist_all(adj_list, n_nodes):
    res = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        res[i], _ = _BFS_dist(adj_list, n_nodes, i)
    return res

def _adj_mat_to_flat_edge(adj_mat):
    if isinstance(adj_mat, np.ndarray):
        s, r = np.where(adj_mat.astype(bool))
    else:
        print(f'tobe implemented _adj_mat_to_flat_edge, type : {type(adj_mat)}')
        exit(1)
    return np.array([s, r])

def pool_edge(g, idx, num_nodes):
    # g in scipy sparse mat
    g = g.toarray()
    g = _adj_mat_to_flat_edge(g)  # now flat edge list
    # idx is list
    idx = np.array(idx, dtype=np.longlong)
    idx_new_valid = np.arange(len(idx)).astype(np.longlong)
    idx_new_all = -1 * np.ones(num_nodes).astype(np.longlong)
    idx_new_all[idx] = idx_new_valid
    new_g = -1 * np.ones_like(g).astype(np.longlong)
    new_g[0] = idx_new_all[g[0]]
    new_g[1] = idx_new_all[g[1]]
    both_valid = np.logical_and(new_g[0] >= 0, new_g[1] >= 0)
    e_idx = np.where(both_valid)[0]
    new_g = new_g[:, e_idx]

    return new_g, e_idx

def _min_ave_seed(adj_list, clusters):
    seeds = []
    dist = _BFS_dist_all(adj_list, len(adj_list))
    for c in clusters:
        d_c = dist[c]
        d_c = d_c[:, c]
        d_sum = np.sum(d_c, axis=1)
        min_ave_depth_node = c[np.argmin(d_sum)]
        seeds.append(min_ave_depth_node)

    return seeds


def triangles_to_edges(cells):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    t_cell = torch.tensor(cells)
    edge_index = torch.cat((t_cell[:, :2], t_cell[:, 1:3], torch.cat((t_cell[:, 2].unsqueeze(1), t_cell[:, 0].unsqueeze(1)), -1)), 0)
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single torch long tensor
    r, _ = torch.min(edge_index, 1, keepdim=True)
    s, _ = torch.max(edge_index, 1, keepdim=True)
    packed_edges = torch.cat((s, r), 1).type(torch.long)
    # remove duplicates and unpack
    unique_edges = torch.unique(packed_edges, dim=0)
    s, r = unique_edges[:, 0], unique_edges[:, 1]
    # create two-way connectivity
    return torch.stack((torch.cat((s, r), 0), torch.cat((r, s), 0))).numpy()

def _find_clusters(adj_list, mask=None):
    n_nodes = len(adj_list)
    if isinstance(mask, list):
        remaining_nodes = []
        for i, m in enumerate(mask):
            if m == True:
                remaining_nodes.append(i)
    else:
        remaining_nodes = list(range(n_nodes))
    cluster = []
    while remaining_nodes:
        if len(remaining_nodes) > 1:
            seed = remaining_nodes[0]
            dist, _ = _BFS_dist(adj_list, n_nodes, seed, mask)
            tmp = []
            new_remaining = []
            for n in remaining_nodes:
                if dist[n] != _INF:
                    tmp.append(n)
                else:
                    new_remaining.append(n)
            cluster.append(tmp)
            remaining_nodes = new_remaining
        else:
            cluster.append([remaining_nodes[0]])
            break

    return cluster

def bstride_selection(flat_edge, n_nodes, pos_mesh = None):
    combined_idx_kept = set()

    #####_flat_edge_to_adj_list:
    # adj_list holds all connection between nodes.
    adj_list = [[] for _ in range(n_nodes)]
    for i in range(len(flat_edge[0])):
        adj_list[flat_edge[0, i]].append(flat_edge[1, i])


    #####_flat_edge_to_adj_mat:
    adj_mat = scipy.sparse.coo_array((np.ones_like(flat_edge[0]), (flat_edge[0], flat_edge[1])), shape=(n_nodes, n_nodes))
    #####_flat_edge_to_adj_mat:

    # adj mat enhance the diag
    adj_mat.setdiag(1)
    # 0. compute clusters, each of which should be deivded independantly

    #####_find_clusters:
    clusters = _find_clusters(adj_list)
    #####_find_clusters:

    # 1. seeding: by BFS_all for small graphs, or by seed_heuristic for larger graphs
    seeds = _min_ave_seed(adj_list, clusters)
    for seed, c in zip(seeds, clusters):
        n_c = len(c)
        odd = set() 
        even = set()
        index_kept = set()
        dist_from_cental_node, _ = _BFS_dist(adj_list, len(adj_list), seed)
        # Walks through the graph distinguishing nodes between odd and even length from seed node.
        for i in range(len(dist_from_cental_node)):
            if dist_from_cental_node[i] % 2 == 0 and dist_from_cental_node[i] != _INF:
                even.add(i)
            elif dist_from_cental_node[i] % 2 == 1 and dist_from_cental_node[i] != _INF:
                odd.add(i)
        # 4. enforce n//2 candidates
        if len(even) <= len(odd) or len(odd) == 0:
            index_kept = even
            index_rmvd = odd
            delta = len(index_rmvd) - len(index_kept)
        else:
            index_kept = odd
            index_rmvd = even
            delta = len(index_rmvd) - len(index_kept)

        if delta > 0:
            # sort the dist of idx rmvd
            # cal stride based on delta nodes to select
            # generate strided idx from rmvd idx
            # union
            index_rmvd = list(index_rmvd)
            dist_id_rmvd = np.array(dist_from_cental_node)[index_rmvd]
            sort_index = np.argsort(dist_id_rmvd)
            stride = len(index_rmvd) // delta + 1
            delta_idx = sort_index[0::stride]
            delta_idx = set([index_rmvd[i] for i in delta_idx])
            index_kept = index_kept.union(delta_idx)

        combined_idx_kept = combined_idx_kept.union(index_kept)
    # TODO: UNDERSTAND THIS SHIT! 
    combined_idx_kept = list(combined_idx_kept)
    adj_mat = adj_mat.tocsr().astype(float)
    adj_mat = adj_mat@adj_mat
    adj_mat.setdiag(0)
    adj_mat, e_idx = pool_edge(adj_mat, combined_idx_kept, n_nodes)

    return combined_idx_kept, adj_mat, e_idx


def generate_multi_layer_stride(flat_edge, num_l, n, pos_mesh = None):
    m_gs = [flat_edge]
    e_s = []
    m_ids = []
    g = flat_edge
    for l in range(num_l):
        n_l = n if l == 0 else len(index_to_keep)
        index_to_keep, g, e_idx = bstride_selection(g, n_nodes=n_l, pos_mesh=pos_mesh)
        #pos_mesh = pos_mesh[index_to_keep]
        m_gs.append(torch.tensor(g))
        e_s.append(e_idx)
        m_ids.append(index_to_keep)

    return m_gs, m_ids, e_s

# Used in dataset class
""" def _cal_multi_mesh(fields, cells, args):
    mm_dir = os.path.join(args.data_dir, '/mm_files/')
    mmfile = os.path.join(mm_dir, str(args.instance_id) + '_mmesh_layer_' + str(args.layer_num) + '.dat')
    mmexist = os.path.isfile(mmfile)
    if not mmexist:
        edge_i = triangles_to_edges(cells)
        m_gs, m_ids = generate_multi_layer_stride(edge_i,
                                                    args.layer_num,
                                                    n=fields['mesh_pos'].shape[-2],
                                                    pos_mesh=fields["mesh_pos"][0].clone().detach().numpy())
        m_mesh = {'m_gs': m_gs, 'm_ids': m_ids}
        pickle.dump(m_mesh, open(mmfile, 'wb'))
    else:
        m_mesh = pickle.load(open(mmfile, 'rb'))
        m_gs, m_ids = m_mesh['m_gs'], m_mesh['m_ids']
    return m_gs, m_ids """