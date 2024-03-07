#!/usr/bin/env python3
"""Main file for running setup, training and testing"""
import argparse
import os
import sys
import warnings
import copy
import torch
import json

from loguru import logger
from torch_geometric.loader import DataLoader
from sklearn.model_selection import ParameterGrid, train_test_split
from datetime import datetime
from random import choice

sys.path.append('../')
sys.path.append('dataprocessing')
sys.path.append('model')
sys.path.append('utils')
from dataprocessing.dataset import MeshDataset, DatasetPairs
from dataprocessing.utils.loading import save_traj_pairs
from model.model import MultiScaleAutoEncoder
from utils.visualization import make_gif, plot_loss
from utils.parserfuncs import none_or_str, none_or_int, none_or_float, t_or_f
from utils.helperfuncs import build_optimizer, merge_dataset_stats, fetch_random_args
from train import test, train

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
parser.add_argument('-ae_layers', type=int, default=2)
parser.add_argument('-alpha', type=float, default=0.5)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-args_file', type=none_or_str, default=None)
parser.add_argument('-data_dir', type=str, default='../data/cylinder_flow/trajectories_1768')
parser.add_argument('-dual_loss', type=t_or_f, default = False)
parser.add_argument('-epochs', type=int, default=1)
parser.add_argument('-edge_conv', type=t_or_f, default=True)
parser.add_argument('-hidden_dim', type=int, default=32)
parser.add_argument('-instance_id', type=int, default=1)
parser.add_argument('-latent_space', type=t_or_f, default=True)
parser.add_argument('-logger_lvl', type=str, default='INFO')
parser.add_argument('-loss', type=none_or_str, default='LMSE')
parser.add_argument('-masked_loss', type=t_or_f, default = True)
parser.add_argument('-load_model', type=t_or_f, default=False)
parser.add_argument('-loss_step', type=int, default=10)
parser.add_argument('-log_step', type=int, default=10)
parser.add_argument('-latent_dim', type=int, default=64)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-make_gif', type=t_or_f, default=False)
parser.add_argument('-max_latent_nodes', type=int, default = 0)
parser.add_argument('-max_latent_edges', type=int, default = 0)
parser.add_argument('-model_file', type=str, default="model.pt")
parser.add_argument('-mpl_ratio', type=float, default=0.8)
parser.add_argument('-mpl_layers', type=int, default=1)
parser.add_argument('-normalize', type=t_or_f, default=False)
parser.add_argument('-num_blocks', type=int, default=1)
parser.add_argument('-num_workers', type=int, default=1)
parser.add_argument('-n_nodes', type=int, default=0)
parser.add_argument('-opt', type=str, default='adam')
parser.add_argument('-out_feature_dim', type=none_or_int, default=2)
parser.add_argument('-pool_strat', type=str, default='SAG')
parser.add_argument('-progress_bar', type=t_or_f, default=False)
parser.add_argument('-pretext_task', type=t_or_f, default=False)
parser.add_argument('-random_search', type=t_or_f, default=False)
parser.add_argument('-residual', type=t_or_f, default=True)
parser.add_argument('-save_args_dir', type=str, default='../logs/args/'+day)
parser.add_argument('-save_accuracy_dir', type=str, default='../logs/accuracies/'+day)
parser.add_argument('-graph_structure_dir', type=str, default='../logs/graph_structure/')
parser.add_argument('-save_gif_dir', type=str, default='../logs/gifs/'+day)
parser.add_argument('-save_loss_over_t_dir', type=str, default='../logs/loss_over_t/'+day)
parser.add_argument('-save_mesh_dir', type=str, default='../logs/meshes/'+day)
parser.add_argument('-save_model_dir', type=str, default='../logs/model_chkpoints/'+day)
parser.add_argument('-save_visualize_dir', type=str, default='../logs/visualizations/'+day)
parser.add_argument('-shuffle', type=t_or_f, default=True)

parser.add_argument('-save_plot', type=t_or_f, default=True)
parser.add_argument('-save_model', type=t_or_f, default=True)
parser.add_argument('-save_latent', type=t_or_f, default=False)
parser.add_argument('-save_visual', type=t_or_f, default=True)
parser.add_argument('-save_losses', type=t_or_f, default=True)
parser.add_argument('-save_mesh', type=t_or_f, default=True)
parser.add_argument('-save_plot_dir', type=str, default='../logs/plots/'+day)
parser.add_argument('-train', type=t_or_f, default=True)
parser.add_argument('-transform', type=t_or_f, default=False)
parser.add_argument('-transform_p', type=float, default=0.3)
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
    logger.debug(f'SAVE_MODEL : {args.save_model_dir}')
    if args.args_file is not None:
        if os.path.isfile(args.args_file):
            with open(args.args_file, 'r') as f:
                args_dict = json.loads(f.read())
                ignored_keys = ['load_model', 'make_gif', 'device', 'model_file', 'args_file', 'random_search']
                for k, v in args_dict.items():
                    if k in ignored_keys:
                        continue
                    logger.info(f'{k} : {v}')
                    args.__dict__[k] = v
                logger.success(f'Args loaded from {args.args_file}')

    # Initialize dataset, containing one trajecotry.
    # NOTE: This will be changed to only take <args>
    train_data = MeshDataset(args=args, mode = 'train')
    test_data = MeshDataset(args=args, mode = 'test')
    val_data = MeshDataset(args=args, mode = 'val')
    args.in_dim_node, args.in_dim_edge, args.n_nodes = (
        train_data[0].num_features,
        train_data[0].edge_attr.shape[1],
        train_data[0].x.shape[0]
    )
    (m_ids, m_gs, 
    e_s, m_pos, 
    args.max_latent_nodes, 
    args.max_latent_edges, 
    graph_placeholders) = merge_dataset_stats(train_data, test_data, val_data)
    # Initialize Model
    logger.info(f'{val_data[0]}')
    
    model = MultiScaleAutoEncoder(args, m_ids, m_gs, e_s, m_pos, graph_placeholders)
    logger.success(f'Device {args.device}')
    model = model.to(args.device)
    if args.load_model:
        logger.info("Loading model")
        try:
            model.load_state_dict(torch.load(args.model_file, map_location=args.device))
            logger.success(f"Multi Scale Autoencoder loaded from {args.model_file}")
        except e:
            logger.error(f'Unable to load model from {args.model_file}, because {e}')

    if args.make_gif:
        logger.success('Making a gif <3')
        gif_data = copy.deepcopy(val_data)
        args.model_file = datetime.now().strftime("%d-%m-%y")
        make_gif(model, gif_data, args)
        logger.success('Made a gif <3')
        exit()

    # Initialize optimizer and scheduler(?)
    optimizer = build_optimizer(args, model.parameters())

    # dataset = copy.deepcopy(val_data[:250]) # The rest of the dataset have little variance
    # ================================
    # SPLIT DATASET INTO TEST AND TRAIN
    # ================================
    # dataset = copy.deepcopy(val_data)
    # train_data, test_data = train_test_split(dataset, test_size=args.test_ratio)

    # # Split training data into train and validation data
    # train_data, val_data = train_test_split(
    #    train_data, test_size=args.val_ratio / (1 - args.test_ratio)
    # )
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
    logger.success(f'All data loaded')
    # TRAINING
    with torch.autograd.set_detect_anomaly(True):
        train_losses, val_losses, model = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            args=args,
        )
    if args.save_latent:
        pairs = 'data/cylinder_flow/pairs'
        save_traj_pairs(args.instance_id)
        dataset_pairs = DatasetPairs(args = args)
        encoder = model.encoder.to(args.device)
        encoder_loader = DataLoader(dataset_pairs, batch_size = 1)
        latent_space_path = os.path.join('..','data','latent_space')
        pair_list = []
        pair_list_file = os.path.join(f'{latent_space_path}', f'encoded_dataset_pairs.pt')
        for idx, (graph1, graph2) in enumerate(encoder_loader):
            logger.debug(idx)
            _, z1, _ = encoder(graph1.to(args.device))
            _, z2, _ = encoder(graph2.to(args.device))
            if os.path.isfile(pair_list_file):
                pair_list = torch.load(pair_list_file)
            
            pair_list.append((torch.squeeze(z1, dim = 0), torch.squeeze(z2, dim = 0)))
                
            torch.save(pair_list, pair_list_file)
            # deleting to save memory
            del pair_list
    
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
        # test_loss, loss_over_t, ts = test(model=model, test_loader=test_loader, args=args)
        # loss_name = "loss-over-t_" + args.time_stamp
        # PATH = os.path.join(args.save_loss_over_t_dir, f"{loss_name}.png")
        # plot_test_loss(loss_over_t, ts, PATH=PATH)
if __name__ == "__main__":
    warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")
    logger.remove(0)

    logger.add(sys.stderr, level=args.logger_lvl)

    logger.info(f"CUDA is available: {torch.cuda.is_available()}")
    logger.info(f"CUDA has version: {torch.version.cuda}")

    if args.logger_lvl == 'DEBUG':
        args_string = ""
        for ele in list(vars(args).items()):
            args_string += f"\n{ele}"
        logger.debug(f"args are the following:{args_string}")
    if not args.random_search:
        if args.transform:
            # apply the transform to the data and run the model
            args = apply_transform(args)
        # run the model with the applied args
        main()

    else:
        param_grid = {
            "mpl_layers" : [1, 2, 3],
            "latent_dim": [128, 512],
            "ae_layers": [3, 4, 5],
            "lr" : [1e-4, 1e-5],
            "mpl_ratio" : [0.3, 0.6],
            "loss" : ["MSE", "LMSE"]
        }
        lst = list(ParameterGrid(param_grid))

        my_bool = args.pretext_task

        while len(lst) > 0:
            args, lst = fetch_random_args(args, lst)
            if args.pretext_task:
                args = apply_transform(args)
            logger.info(f"Doing the following config: {args.time_stamp}")
            try:
                main()
            except (ValueError, RuntimeError) as error:
                print(error)
                continue
            logger.success("Done")
            args.pretext_task = my_bool
    logger.success("process_done")
