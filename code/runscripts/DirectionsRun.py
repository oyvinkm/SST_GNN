#!/usr/bin/env python3
"""Main file for running setup, training and testing"""
import argparse
import sys
import warnings
from datetime import datetime

sys.path.append("../")
import os

import torch
from dataprocessing.utils.loading import split_pairs
from latent_trainer import train
from loguru import logger
from model.deformator import LatentDeformator, LatentScaler
from torch_geometric.loader import DataLoader
from utils.helperfuncs import load_args
from utils.parserfuncs import t_or_f
from utils.visualization import deformater_visualize

parser = argparse.ArgumentParser()
now = datetime.now().strftime("%H.%M")
date = datetime.now().strftime("%d_%m")
timestamp = datetime.now().strftime("%d_%m-%H.%M")
parser.add_argument(
    "-decoder_path", type=str, default="../logs/model_chkpoints/decoder.pt"
)
parser.add_argument("-device", type=str, default="cpu")
parser.add_argument("-deformator_type", type=str, default="proj")
parser.add_argument("-epochs", type=int, default=2000)
parser.add_argument("-logger_lvl", type=str, default="DEBUG")
parser.add_argument("-make_gif", type=t_or_f, default="False")
parser.add_argument("-decode_test", type=t_or_f, default="True")
parser.add_argument("-time_of_the_day", type=str, default=now)
parser.add_argument("-time_stamp", type=str, default=timestamp)
parser.add_argument("-vgae_args_path", type=str, default="../logs/args/SOTAargs.json")
parser.add_argument("-save_mesh_dir", type=str, default="../logs/direction/meshes/")
parser.add_argument("-date", type=str, default=date)

deformator_args = parser.parse_args()


def main():
    vgae_args = load_args(path=deformator_args.vgae_args_path)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logger.remove(0)
    logger.add(sys.stderr, level=deformator_args.logger_lvl)
    vgae_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    deformator_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    deformator = LatentDeformator(
        shift_dim=vgae_args.latent_dim,
        input_dim=vgae_args.latent_dim,
        out_dim=vgae_args.latent_dim,
    ).to(vgae_args.device)

    scaler = LatentScaler(vgae_args.latent_dim)

    dataset_pairs = torch.load(
        os.path.join("..", "data", "latent_space", "encoded_dataset_pairs.pt")
    )
    train_set, validation_set = split_pairs(dataset_pairs)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=vgae_args.shuffle)

    validation_loader = DataLoader(validation_set, batch_size=1)

    train(deformator, scaler, train_loader, validation_loader, deformator_args)

    if deformator_args.make_gif:
        deformater_visualize(deformator, train_loader, deformator_args, vgae_args)


if __name__ == "__main__":
    main()
