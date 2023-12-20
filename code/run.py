#!/usr/bin/env python3
"""Main file for running setup, training and testing"""
import argparse
import math
import os
import random
import sys
import warnings
from datetime import datetime
import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader
sys.path.append('Model')
sys.path.append('utils')
from dataprocessing.dataset import MeshDataset
from utils.visualization import plot_dual_mesh, make_gif, plot_test_loss
from Model.model import MultiScaleAutoEncoder
from utils.opt import build_optimizer
from train import test, train
from utils.visualization import plot_loss
from sklearn.model_selection import ParameterGrid
from random import choice

def none_or_str(value):
    if value.lower() == "none":
        return None
    return value


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def none_or_float(value):
    if value == "None":
        return None
    return float(value)
    
def t_or_f(value):
    ua = str(value).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       logger.CRITICAL("boolean argument incorrect")

def apply_transform(args):
    logger.info("Applying Transformation")
    args.time_stamp += "_transform"
    main() 
    logger.info("transform done")
    args.load_model = True
    args.model_file = f"model_{args.time_stamp}.pt"
    args.transform = False
    args.time_stamp += "_post_transform"
    return args

day = datetime.now().strftime("%d-%m-%y")
parser = argparse.ArgumentParser()
parser.add_argument('-ae_ratio', type=none_or_float, default=0.5)
parser.add_argument('-ae_layers', type=int, default=3)
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-data_dir', type=str, default='data/cylinder_flow/')
parser.add_argument('-epochs', type=int, default=101)
parser.add_argument('-edge_conv', type=t_or_f, default=False)
parser.add_argument('-hidden_dim', type=int, default=32)
parser.add_argument('-instance_id', type=int, default=1)
parser.add_argument('-latent_space', type=t_or_f, default=True)
parser.add_argument('-logger_lvl', type=str, default='DEBUG')
parser.add_argument('-loss', type=none_or_str, default='LMSE')
parser.add_argument('-load_model', type=t_or_f, default=False)
parser.add_argument('-loss_step', type=int, default=20)
parser.add_argument('-log_step', type=int, default=20)
parser.add_argument('-latent_dim', type=int, default=128)
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-make_gif', type=t_or_f, default=False)
parser.add_argument('-model_file', type=str, default="model_2023_11_27-13.33_ae_layers-3_hidden_dim-64_latent_dim-128_pool_strat-SAG.pt")
parser.add_argument('-mpl_ratio', type=float, default=0.8)
parser.add_argument('-mpl_layers', type=int, default=1)
parser.add_argument('-normalize', type=t_or_f, default=False)
parser.add_argument('-num_blocks', type=int, default=1)
parser.add_argument('-num_workers', type=int, default=1)
parser.add_argument('-n_nodes', type=int, default=1876)
parser.add_argument('-opt', type=str, default='adam')
parser.add_argument('-out_feature_dim', type=none_or_int, default=11)
parser.add_argument('-pool_strat', type=str, default='ASA')
parser.add_argument('-progress_bar', type=t_or_f, default=False)
parser.add_argument('-random_search', type=t_or_f, default=False)
parser.add_argument('-residual', type=t_or_f, default=True)
parser.add_argument('-save_args_dir', type=str, default='logs/args/'+day)
parser.add_argument('-save_visualize_dir', type=str, default='logs/visualizations/'+day)
parser.add_argument('-save_mesh_dir', type=str, default='logs/meshes/'+day)
parser.add_argument('-save_accuracy_dir', type=str, default='logs/accuracies/'+day)
parser.add_argument('-save_model_dir', type=str, default='logs/model_chkpoints/'+day)
parser.add_argument('-save_gif_dir', type=str, default='logs/gifs/'+day)
parser.add_argument('-save_loss_over_t_dir', type=str, default='logs/loss_over_t/'+day)
parser.add_argument('-shuffle', type=t_or_f, default=True)
parser.add_argument('-save_plot', type=t_or_f, default=True)
parser.add_argument('-save_model', type=t_or_f, default=True)
parser.add_argument('-save_visual', type=t_or_f, default=True)
parser.add_argument('-save_losses', type=t_or_f, default=True)
parser.add_argument('-save_mesh', type=t_or_f, default=True)
parser.add_argument('-save_plot_dir', type=str, default='plots/'+day)
parser.add_argument('-transform', type=t_or_f, default=True)
parser.add_argument('-transform_p', type=float, default=0.1)
parser.add_argument('-time_stamp', type=none_or_str, default=datetime.now().strftime("%Y_%m_%d-%H.%M"))
parser.add_argument('-test_ratio', type=float, default=0.2)
parser.add_argument('-val_ratio', type=float, default=0.1)
parser.add_argument('-weight_decay', type=float, default=0.0005)
args = parser.parse_args()

def main():
    # args.transform = 'Attribute'
    # To ensure reproducibility the best we can, here we control the sources of
    # randomness by seeding the various random number generators used in this Colab
    # For more information, see:
    # https://pytorch.org/docs/stable/notes/randomness.html
    # torch.manual_seed(5)  # Torch
    # random.seed(5)  # Python
    # np.random.seed(5)  # NumPy

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
    # args.latent_vec_dim = math.ceil(dataset[0].num_nodes*(args.ae_ratio**args.ae_layers))
    # Initialize Model
    if not args.latent_space:
        logger.warning("Model is not going into latent_space")
    model = MultiScaleAutoEncoder(args, dataset.m_ids, dataset.m_gs, dataset.e_s)
    model = model.to(args.device)
    if args.load_model:
        model_path = os.path.join(args.save_model_dir , args.model_file)
        logger.info("Loading model")
        logger.debug(f"{model_path}")
        assert os.path.isfile(model_path), "model file does not exist"
        model.load_state_dict(torch.load(model_path))
        logger.success(f"Multi Scale Autoencoder loaded from {args.model_file}")
    
    if args.make_gif:
        make_gif(model, dataset, args)
        exit()

    # Initialize optimizer and scheduler(?)
    optimizer = build_optimizer(args, model.parameters())

    dataset = dataset[:250] # The rest of the dataset have little variation

    # Split data into train and test
    train_data, test_data = train_test_split(dataset, test_size=args.test_ratio)

    # Split training data into train and validation data
    train_data, val_data = train_test_split(
        train_data, test_size=args.val_ratio / (1 - args.test_ratio)
    )
    logger.info(
        f"\n\tTrain size : {len(train_data)}, \n\
        Validation size : {len(val_data)}, \n\
        Test size : {len(test_data)}"
    )
    # Create Dataloaders for train, test and validation
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=args.shuffle
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

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
        if not os.path.isdir(args.save_loss_over_t_dir):
            os.mkdir(args.save_loss_over_t_dir)
        PATH = os.path.join(args.save_plot_dir, f"{loss_name}.png")
        plot_loss(
            train_loss=train_losses,
            train_label="Training Loss",
            val_loss=val_losses,
            val_label="Validation Loss",
            PATH=PATH,
        )
        test_loss, loss_over_t, ts = test(model=model, test_loader=test_loader, args=args)
        loss_name = "loss-over-t_" + args.time_stamp
        PATH = os.path.join(args.save_loss_over_t_dir, f"{loss_name}.png")
        plot_test_loss(loss_over_t, ts, PATH=PATH)


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
    logger.add(sys.stderr, level=args.logger_lvl)

    logger.info(f"CUDA is available: {torch.cuda.is_available()}")
    logger.info(f"CUDA has version: {torch.version.cuda}")
    if not args.random_search:
        if args.transform:
            # apply the transform to the data and run the model
            args = apply_transform(args)
        # run the model with the applied args
        main()

    else:
        param_grid = {
                'hidden_dim' : [8, 16, 32],
                'latent_dim': [64, 128, 256], 
                'ae_layers': [2, 3, 4],
                'num_blocks': [1, 2, 3],
                'pool_strat' : ['SAG', 'ASA'],
                }
        lst = list(ParameterGrid(param_grid))
        my_bool = args.transform

        while True:
            rand_args = choice(lst)
            lst.remove(rand_args)
            args.time_stamp = datetime.now().strftime("%Y_%m_%d-%H.%M")
            for key in rand_args.keys():
                args.__dict__[key] = rand_args[key]
                args.time_stamp += "_" + key + "-" + str(rand_args[key])
            if args.transform:
                args = apply_transform(args)
            logger.success(f"Doing the following config: {args.time_stamp}")
            main()
            logger.success("Done")
            args.transform = my_bool
    logger.success("process_done")