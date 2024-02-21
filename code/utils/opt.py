""" Module just to build optimizer"""
from torch import optim


def build_optimizer(args, params):
    filter_fn = filter(lambda p: p.requires_grad, params)
    optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=0.0005)
    return optimizer
