import argparse
import json
import os
from datetime import datetime
from random import randint

import torch
from dataprocessing.dataset import DatasetPairs
from dataprocessing.utils.loading import save_traj_pairs
from loguru import logger
from torch_geometric.loader import DataLoader
from utils.visualization import plot_loss


@torch.no_grad()
def save_pair_encodings(args, encoder):
    logger.success("Encoding graph pairs with current model...")

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
        f"{latent_space_path}", f"encoded_{dataset_name}set.pt"
    )
    if os.path.exists(pair_list_file):
        os.remove(pair_list_file)
    loader = DataLoader(dataset, batch_size=1)
    for idx, graph in enumerate(loader):
        _, z, _ = encoder(graph.to(args.device))

        if os.path.isfile(pair_list_file):
            pair_list = torch.load(pair_list_file)
        pair_list.append(torch.squeeze(z, dim=0))
        torch.save(pair_list, pair_list_file)

        # deleting to save memory
        del pair_list
    logger.success("Encoding done...")


def load_model(args, model):
    load_model(args)
    model_path = os.path.join(args.save_model_dir, args.model_file)
    logger.info("Loading model")
    assert os.path.isfile(model_path), "model file does not exist"
    model.load_state_dict(torch.load(model_path))
    logger.success(f"Multi Scale Autoencoder loaded from {args.model_file}")


def save_plot(args, model, train_losses, validation_losses):
    loss_name = "loss_" + args.time_stamp
    if not os.path.isdir(args.save_plot_dir):
        os.mkdir(args.save_plot_dir)
    if not os.path.isdir(args.save_loss_over_t_dir):
        os.mkdir(args.save_loss_over_t_dir)
    PATH = os.path.join(args.save_plot_dir, f"{loss_name}.png")
    plot_loss(
        train_loss=train_losses,
        train_label="Training Loss",
        validation_loss=validation_losses,
        val_label="Validation Loss",
        PATH=PATH,
    )


def load_args(path):
    """loads the args of the VGAE"""
    args = argparse.Namespace()
    with open(path) as json_file:
        d = json.load(json_file)
        for key, val in d.items():
            args.__dict__[key] = val
    return args


def print_args(args):
    args_string = ""
    for ele in list(vars(args).items()):
        args_string += f"\n{ele}"
    logger.debug(f"args are the following:{args_string}")


def fetch_random_args(args, lst):
    rand_number = randint(0, len(lst) - 1)
    rand_args = lst[rand_number]
    lst.remove(rand_args)
    args.time_stamp = datetime.now().strftime("%Y_%m_%d-%H.%M")
    for key in rand_args.keys():
        args.__dict__[key] = rand_args[key]
        args.time_stamp += "_" + key + "-" + str(rand_args[key])

    return args, lst


def save_bi_stride_dir(args, dataset):
    path = args.save_graph_structure_dir
    if not os.path.isdir(path):
        os.mkdir(path)
    path_id = os.path.join(args.save_graph_structure_dir, f"{args.instance_id}")
    if not os.path.isdir(path_id):
        os.mkdir(path_id)
    path_id_ae = args.graph_structure_aelayers_dir
    os.mkdir(path_id_ae)
    torch.save(dataset.m_ids, os.path.join(path_id_ae, "m_ids.pt"))
    torch.save(dataset.m_gs, os.path.join(path_id_ae, "m_gs.pt"))
    torch.save(dataset.e_s, os.path.join(path_id_ae, "e_s.pt"))


def get_bi_stride_structure(args):
    m_ids = torch.load(os.path.join(args.graph_structure_aelayers_dir, "m_ids.pt"))
    m_gs = torch.load(os.path.join(args.graph_structure_aelayers_dir, "m_gs.pt"))
    e_s = torch.load(os.path.join(args.graph_structure_aelayers_dir, "e_s.pt"))

    return m_ids, m_gs, e_s
