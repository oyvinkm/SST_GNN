""" Module just to build optimizer"""
from torch import optim

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    return optimizer
