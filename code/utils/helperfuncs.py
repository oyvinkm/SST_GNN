import json
import os
from datetime import datetime
from random import randint

import torch
from dataprocessing.dataset import DatasetPairs
from dataprocessing.utils.loading import save_traj_pairs
from loguru import logger
from torch import optim
from torch_geometric.loader import DataLoader
from utils.visualization import plot_loss, save_mesh

@torch.no_grad()
def save_pictures(model, data, args):
    """
    Performs a validation run on our current model with the validationset
    saved in the val_loader.
    """
    total_loss = 0
    model.eval()

    logger.debug("======= SAVING PIX =======")
    early = False
    late = False
    for idx, batch in enumerate(data):
        if idx % 10 == 0:
            batch = batch.to(args.device)
            # batch.x = F.normalize(batch.x)
            b_data = batch.clone()
            pred, _ = model(b_data, Train=False)
            save_mesh(pred, batch, f"test_{batch.t.item()}", args)

@torch.no_grad()
def save_pair_encodings(args, encoder):
    logger.info("encoding graph pairs with current model...")

    save_traj_pairs(args.instance_id)
    dataset_pairs = DatasetPairs(args=args)
    encoder_loader = DataLoader(dataset_pairs, batch_size=1)
    latent_space_path = os.path.join("..", "data", "latent_space")
    pair_list = []
    pair_list_file = os.path.join(f"{latent_space_path}", "encoded_dataset_pairs.pt")
    if os.path.exists(pair_list_file):
        os.remove(pair_list_file)
    for idx, (graph1, graph2) in enumerate(encoder_loader):
        _, z1, _ = encoder(graph1.to(args.device))
        _, z2, _ = encoder(graph2.to(args.device))
        if os.path.isfile(pair_list_file):
            pair_list = torch.load(pair_list_file)
        pair_list.append((torch.squeeze(z1, dim=0), torch.squeeze(z2, dim=0)))
        torch.save(pair_list, pair_list_file)

        # deleting to save memory
        del pair_list
    logger.success("Encoding done...")


@torch.no_grad()
def encode_and_save_set(args, encoder, dataset):
    logger.info("encoding graphs from  with current model...")
    latent_space_path = os.path.join("..", "data", "latent_space")
    pair_list = []
    dataset_name = f"{dataset=}".split("=")[0].split("_")[0]
    pair_list_file = os.path.join(
        f"{latent_space_path}", f"encoded_{dataset_name}_{args.time_stamp}.pt"
    )
    if os.path.exists(pair_list_file):
        os.remove(pair_list_file)
    loader = DataLoader(dataset, batch_size=1)
    for idx, graph in enumerate(loader):
        _, z, _ = encoder(graph.to(args.device))

        if os.path.isfile(pair_list_file):
            pair_list = torch.load(pair_list_file)
        z.t = graph.t
        pair_list.append(z)
        torch.save(pair_list, pair_list_file)

        # deleting to save memory
        del pair_list
    logger.success(f'Encodings saved at {pair_list_file}')
    logger.success("Encoding done...")


def load_model(args, model):
    logger.info("Loading model")
    assert os.path.isfile(
        args.model_file
    ), f"model file {args.model_file} does not exist"
    model.load_state_dict(torch.load(args.model_file, map_location=args.device))
    logger.success(f"Multi Scale Autoencoder loaded from {args.model_file}")
    logger.success("TEST: Can we remove the return statement on next line?")
    return model


def load_args(args):
    """loads the args of the VGAE"""
    if not args.load_model:
        return 0
    str_splt = args.model_file.split("/")
    args_file = os.path.join(
        str_splt[0],
        str_splt[1],
        "args",
        str_splt[3],
        "args_" + str_splt[4][6:] + ".json",
    )
    assert os.path.isfile(args_file), f"{args_file=} doesn't exist"
    logger.info("Loading args since we are loading a model...")
    with open(args_file, "r") as f:
        args_dict = json.loads(f.read())
        ignored_keys = [
            "load_model",
            "make_gif",
            "device",
            "model_file",
            "args_file",
            "random_search",
            "time_stamp",
            "save_encodings",
            "make_pics"
        ]
        for k, v in args_dict.items():
            if k in ignored_keys:
                continue
            logger.info(f"{k} : {v}")
            args.__dict__[k] = v
        logger.success(f"Args loaded from {args_file}")
        return args


def print_args(args):
    args_string = ""
    for ele in list(vars(args).items()):
        args_string += f"\n{ele}"
    logger.debug(f"args are the following:{args_string}")


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    return optimizer


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def merge_dataset_stats(train, val, test):
    graph_placeholders = Merge(
        Merge(train.graph_placeholders, test.graph_placeholders), val.graph_placeholders
    )
    train_id, train_g, train_e, train_pos = train._get_pool()
    test_id, test_g, test_e, test_pos = test._get_pool()
    val_id, val_g, val_e, val_pos = val._get_pool()
    m_ids, m_gs, e_s, m_pos = [], [], [], []
    max_latent_nodes = max(
        [train.max_latent_nodes, test.max_latent_nodes, val.max_latent_nodes]
    )
    max_latent_edges = max(
        [train.max_latent_edges, test.max_latent_edges, val.max_latent_edges]
    )
    for i in range(len(train_id)):
        m_ids.append(Merge(Merge(train_id[i], test_id[i]), val_id[i]))
        e_s.append(Merge(Merge(train_e[i], test_e[i]), val_e[i]))
    for i in range(len(train_g)):
        m_gs.append(Merge(Merge(train_g[i], test_g[i]), val_g[i]))
        m_pos.append(Merge(Merge(train_pos[i], test_pos[i]), val_pos[i]))
    return (
        m_ids,
        m_gs,
        e_s,
        m_pos,
        max_latent_nodes,
        max_latent_edges,
        graph_placeholders,
    )


def fetch_random_args(args, lst):
    rand_number = randint(0, len(lst) - 1)
    rand_args = lst[rand_number]
    lst.remove(rand_args)
    args.time_stamp = datetime.now().strftime("%Y_%m_%d-%H.%M")
    for key in rand_args.keys():
        args.__dict__[key] = rand_args[key]
        args.time_stamp += "_" + key + "-" + str(rand_args[key])
    logger.success(f"Doing the following config: {args.time_stamp}")

    return args, lst


def save_plot(args, train_losses, validation_losses):
    loss_name = "loss_" + args.time_stamp
    if not os.path.isdir(args.save_plot_dir):
        os.mkdir(args.save_plot_dir)
    if not os.path.isdir(args.save_loss_over_t_dir):
        os.mkdir(args.save_loss_over_t_dir)
    PATH = os.path.join(args.save_plot_dir, f"{loss_name}.png")
    plot_loss(
        train_loss=train_losses,
        train_label="Training Loss",
        validation_losses=validation_losses,
        val_label="Validation Loss",
        PATH=PATH,
    )
