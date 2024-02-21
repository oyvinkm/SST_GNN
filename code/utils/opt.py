""" Module just to build optimizer"""
from torch import optim

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    return optimizer


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res
def merge_dataset_stats(train, val, test):
    train_id, train_g, train_e = train._get_pool()
    test_id, test_g, test_e = test._get_pool()
    val_id, val_g, val_e = val._get_pool()
    m_ids, m_gs, e_s = [], [], []
    max_latent_nodes = max([train.max_latent_nodes, 
                                 test.max_latent_nodes, 
                                 val.max_latent_nodes])
    max_latent_edges = max([train.max_latent_edges, 
                                 test.max_latent_edges, 
                                 val.max_latent_edges])
    for i in range(len(train_id)):
        m_ids.append(Merge(Merge(train_id[i], test_id[i]), val_id[i])) 
    for i in range(len(train_g)):
        m_gs.append(Merge(Merge(train_g[i], test_g[i]), val_g[i]))
    for i in range(len(train_e)):
        e_s.append(Merge(Merge(train_e[i], test_e[i]), val_e[i]))
    return m_ids, m_gs, e_s, max_latent_nodes, max_latent_edges