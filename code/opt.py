""" Module just to build optimizer"""
from torch import optim


def build_optimizer(args, params):
    """
    Builds optimizer depending on the parameters and arguments.
    """
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    scheduler = None
    if args.opt == "adam":
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(
            filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay
        )
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "adagrad":
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler is None:
        return None, optimizer
    if args.opt_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate
        )
    elif args.opt_scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart
        )
    return scheduler, optimizer