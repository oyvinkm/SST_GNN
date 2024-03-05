""" Module just to build optimizer"""
from datetime import datetime
from torch import optim
from random import randint


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    return optimizer


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def merge_dataset_stats(train, val, test):
    graph_placeholders = Merge(Merge(train.graph_placeholders, 
                                    test.graph_placeholders), 
                                    val.graph_placeholders)
    train_id, train_g, train_e, train_pos= train._get_pool()
    test_id, test_g, test_e, test_pos = test._get_pool()
    val_id, val_g, val_e, val_pos = val._get_pool()
    m_ids, m_gs, e_s, m_pos = [], [], [], []
    max_latent_nodes = max([train.max_latent_nodes, 
                                 test.max_latent_nodes, 
                                 val.max_latent_nodes])
    max_latent_edges = max([train.max_latent_edges, 
                                 test.max_latent_edges, 
                                 val.max_latent_edges])
    for i in range(len(train_id)):
        m_ids.append(Merge(Merge(train_id[i], test_id[i]), val_id[i])) 
        e_s.append(Merge(Merge(train_e[i], test_e[i]), val_e[i]))
    for i in range(len(train_g)):
        m_gs.append(Merge(Merge(train_g[i], test_g[i]), val_g[i]))
        m_pos.append(Merge(Merge(train_pos[i], test_pos[i]), val_pos[i]))
    return m_ids, m_gs, e_s, m_pos, max_latent_nodes, max_latent_edges, graph_placeholders


def fetch_random_args(args, lst):
    rand_number = randint(0, len(lst) - 1)
    rand_args = lst[rand_number]
    lst.remove(rand_args)
    args.time_stamp = datetime.now().strftime("%Y_%m_%d-%H.%M")
    for key in rand_args.keys():
        args.__dict__[key] = rand_args[key]
        args.time_stamp += "_" + key + "-" + str(rand_args[key])

    return args, lst