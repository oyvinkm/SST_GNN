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


def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        logger.info(f"created folder at {path}")


def save_graph_structure(args, m_ids, m_gs, e_s, m_pos, graph_placeholders):
    create_folder(args.graph_structure_dir)
    PATH = os.path.join(args.graph_structure_dir, f"{args.instance_id}")
    create_folder(PATH)
    PATH = os.path.join(PATH, f"{args.ae_layers}")
    create_folder(PATH)
    # These five save operations took 0.15 seconds, thus it's okay we override them everytime
    torch.save(m_ids, os.path.join(PATH, "m_ids.pt"))
    torch.save(m_gs, os.path.join(PATH, "m_gs.pt"))
    torch.save(e_s, os.path.join(PATH, "e_s.pt"))
    torch.save(m_pos, os.path.join(PATH, "m_pos.pt"))
    torch.save(graph_placeholders, os.path.join(PATH, "graph_placeholders.pt"))


def create_encodings_folders(args):
    model_file_splt = args.model_file.split("/")

    latent_space_path = os.path.join("..", "data", "latent_space")
    if not os.path.isdir(latent_space_path):
        os.mkdir(latent_space_path)

    day_folder = model_file_splt[3]
    PATH = os.path.join(latent_space_path, day_folder)
    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    model_folder = model_file_splt[4]
    PATH = os.path.join(PATH, model_folder)
    if not os.path.isdir(PATH):
        os.mkdir(PATH)

    return PATH


@torch.no_grad()
def save_pair_encodings(args, encoder):
    logger.info("encoding graph pairs with current model...")

    save_traj_pairs(args.instance_id)
    dataset_pairs = DatasetPairs(args=args)
    PATH = create_encodings_folders(args)
    encoder_loader = DataLoader(dataset_pairs, batch_size=1)

    pair_list_file = os.path.join(f"{PATH}", "encoded_dataset_pairs.pt")
    if os.path.exists(pair_list_file):
        os.remove(pair_list_file)
    pair_list = []
    for idx, (graph1, graph2, graph3) in enumerate(encoder_loader):
        _, z1, _ = encoder(graph1.to(args.device))
        _, z2, _ = encoder(graph2.to(args.device))
        _, z3, _ = encoder(graph3.to(args.device))
        if os.path.isfile(pair_list_file):
            pair_list = torch.load(pair_list_file)
        z1.z = torch.squeeze(z1.z, dim=0)
        z2.z = torch.squeeze(z2.z, dim=0)
        z3.z = torch.squeeze(z3.z, dim=0)
        pair_list.append((z1, z2, z3))
        torch.save(pair_list, pair_list_file)

        # deleting to save memory
        del pair_list
    logger.success("Encoding done...")


@torch.no_grad()
def encode_and_save_set(args, encoder, dataset):
    logger.info("encoding graphs from  with current model...")
    PATH = create_encodings_folders(args)
    pair_list_file = os.path.join(PATH, "encoded_dataset.pt")
    if os.path.exists(pair_list_file):
        os.remove(pair_list_file)

    pair_list = []
    loader = DataLoader(dataset, batch_size=1)
    for idx, graph in enumerate(loader):
        _, z, _ = encoder(graph.to(args.device))

        if os.path.isfile(pair_list_file):
            pair_list = torch.load(pair_list_file)
        z.z = torch.squeeze(z.z, dim=0)
        pair_list.append(z)
        torch.save(pair_list, pair_list_file)

        # deleting to save memory
        del pair_list
    logger.success("Encoding done...")


def get_dataset_pairs(args):
    str_splt = args.decoder_path.split("/")
    logger.debug(str_splt)
    PATH = os.path.join(
        str_splt[0],
        "data",
        "latent_space",
        str_splt[3],
        str_splt[4],
        "encoded_dataset_pairs.pt",
    )

    assert os.path.isfile(PATH), f"encoded_dataset_pairs at {PATH=} doesn't exist"
    encoded_dataset_pairs = torch.load(PATH, map_location=torch.device(args.device))
    res = [(x.z, y.z, z.z) for (x, y, z) in encoded_dataset_pairs]
    return res


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
    # I can't be arsed, shitty solution, but i mean... It works every time
    try:
        model_path = args.model_file
    except AttributeError:
        model_path = args.decoder_path

    str_splt = model_path.split("/")
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
        # This defines what the old args should control.
        ignored_keys = [
            "load_model",
            "make_gif",
            "device",
            "model_file",
            "args_file",
            "logger_lvl",
            "random_search",
            "epochs",
            "batch_size",
            "save_mesh_dir",  # make it compatible with directions
            "time_stamp",  # We get a new time_stamp
            "save_encodings",  # The old run shouldn't determine whether we save encodings now
            # "instance_id",  # The old run shouldn't determine whether what trajectory we run, yes it should since the structure of our graph and model depends on this
        ]
        for k, v in args_dict.items():
            if k in ignored_keys:
                continue
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
