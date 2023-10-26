#!/usr/bin/env python3
"""Main file for running setup, training and testing"""
import os
import random
import sys
import warnings
from datetime import datetime

import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric import transforms as T

from dataprocessing.dataset import MeshDataset
from model import MultiScaleAutoEncoder
from mask import AttributeMask
from opt import build_optimizer
from train import test, train
from utils.visualization import plot_loss

# TODO: Set up args so they can be called from config file
# NOTE: Set args up to take loss function.


def main():
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    for args in [
        {
            "data_dir": "data/cylinder_flow/",
            "instance_id": 1,
            "normalize": False,
            "epochs": 20,
            "test_ratio": 0.2,
            "val_ratio": 0.1,
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 1,
            "transforms": None,
            "num_layers": 2,
            "out_feature_dim": 11,
            "ae_layers": 2,
            "ae_ratio": 0.5,
            "in_dim_node": 11,
            "hidden_dim": 64,  # 64
            "in_dim_edge": 3,
            "mpl_layers": 2,  # 2
            "num_blocks": 2,  # 2
            "latent_dim": None,
            "pool_strat": "ASA",
            "mpl_ratio": 0.5,
            "opt": "adam",
            "lr": 0.001,
            "weight_decay": 0.0005,
            "opt_decay_step": 30,
            "time_stamp": None,
            "transform_p" : 0.2,
            "transform" : "node",
            "transforms" : None,
            "opt_scheduler": None,
            "save_model": False,
            "save_plot": True,
            "save_mesh": True,
            "save_model_dir": "model_chkpoints",
            "save_args_dir": "args_chkpoints",
            "save_plot_dir": "plots",
            "save_mesh_dir": "meshes",
        },
    ]:
        args = objectview(args)

    # To ensure reproducibility the best we can, here we control the sources of
    # randomness by seeding the various random number generators used in this Colab
    # For more information, see:
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(5)  # Torch
    random.seed(5)  # Python
    np.random.seed(5)  # NumPy

    # Define the model name for saving
    now = datetime.now()
    args.time_stamp = now.strftime("%d.%m.%Y_%H.%M.%S")

    # Set device to cuda if availale
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    logger.info(f"Device : {args.device}")

    # Initialize dataset, containing one trajecotry.
    # NOTE: This will be changed to only take <args>
    dataset = MeshDataset(args=args)
    args.in_dim_node, args.in_dim_edge = (
        dataset[0].num_features,
        dataset[0].edge_attr.shape[1],
    )

    # Initialize Model
    model = MultiScaleAutoEncoder(args)

    # Initialize optimizer and scheduler(?)
    scheduler, optimizer = build_optimizer(args, model.parameters())

    # Split data into train and test
    train_data, test_data = train_test_split(dataset, test_size=args.test_ratio)

    # Split training data into train and validation data
    # TODO: calculate correct val_ratio
    train_data, val_data = train_test_split(train_data, test_size=args.val_ratio)
    
    logger.info(
        f"Train size : {len(train_data)}, \
        Validation size : {len(val_data)}, \
        Test size : {len(test_data)}"
    )
    # Create Dataloaders for train, test and validation
    # TODO: transform in dataloader
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=args.shuffle
    )
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    if args.transform == 'node':
        args.transforms = T.Compose([AttributeMask(args.transform_p)])
    else: 
        args.transforms = None
        
    # TRAINING

    train_losses, val_losses, model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        args=args,
    )

    if args.save_plot:
        loss_name = "loss_" + args.time_stamp
        if not os.path.isdir(args.save_plot_dir):
            os.mkdir(args.save_plot_dir)
        PATH = os.path.join(args.save_plot_dir, f"{loss_name}.png")
        plot_loss(
            train_loss=train_losses,
            train_label="Training Loss",
            val_loss=val_losses,
            val_label="Validation Loss",
            PATH=PATH,
        )

    test_loss = test(model=model, test_loader=test_loader, loss_func=None, args=args)
    logger.debug(test_loss)


# TESTING
# test()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")
    logger.remove(0)
    # Set the level of what logs to produce, hierarchy:
    # TRACE (5): used to record fine-grained information about the program's
    # execution path for diagnostic purposes.
    # DEBUG (10): used by developers to record messages for debugging purposes.
    # INFO (20): used to record informational messages that describe the normal
    # operation of the program.
    # SUCCESS (25): similar to INFO but used to indicate the success of an operation.
    # WARNING (30): used to indicate an unusual event that may require
    # further investigation.
    # ERROR (40): used to record error conditions that affected a specific operation.
    # CRITICAL (50): used to used to record error conditions that prevent a core
    # function from working.
    logger.add(sys.stderr, level="INFO")

    logger.info(f"CUDA is available: {torch.cuda.is_available()}")
    logger.info(f"CUDA has version: {torch.version.cuda}")  
    main()
