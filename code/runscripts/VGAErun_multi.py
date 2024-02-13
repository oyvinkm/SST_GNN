#!/usr/bin/env python3
"""Main file for running setup, training and testing"""
import argparse
import os
import sys
import warnings
from datetime import datetime
import torch
from loguru import logger
from torch_geometric.loader import DataLoader
from sklearn.model_selection import ParameterGrid
from random import choice
sys.path.append('../')
sys.path.append('dataprocessing')
sys.path.append('model')
sys.path.append('utils')
from dataprocessing.dataset import MeshDataset, DatasetPairs
from dataprocessing.utils.loading import save_traj_pairs
from Model_multi.model import MultiScaleAutoEncoder
from utils.visualization import make_gif, plot_loss
from utils.parserfuncs import none_or_str, none_or_int, none_or_float, t_or_f
from utils.opt import build_optimizer, merge_dataset_stats
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
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-data_dir', type=str, default='../data/cylinder_flow/trajectories_1768')
parser.add_argument('-epochs', type=int, default=1)
parser.add_argument('-edge_conv', type=t_or_f, default=False)
parser.add_argument('-hidden_dim', type=int, default=32)
parser.add_argument('-instance_id', type=int, default=1)
parser.add_argument('-latent_space', type=t_or_f, default=True)
parser.add_argument('-logger_lvl', type=str, default='INFO')
parser.add_argument('-loss', type=none_or_str, default='LMSE')
parser.add_argument('-load_model', type=t_or_f, default=False)
parser.add_argument('-loss_step', type=int, default=10)
parser.add_argument('-log_step', type=int, default=10)
parser.add_argument('-latent_dim', type=int, default=128)
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-make_gif', type=t_or_f, default=False)
parser.add_argument('-max_latent_nodes', type=int, default = 1768)
parser.add_argument('-model_file', type=str, default="sst_gvae.pt")
parser.add_argument('-mpl_ratio', type=float, default=0.8)
parser.add_argument('-mpl_layers', type=int, default=1)
parser.add_argument('-normalize', type=t_or_f, default=False)
parser.add_argument('-num_blocks', type=int, default=1)
parser.add_argument('-num_workers', type=int, default=1)
parser.add_argument('-n_nodes', type=int, default=0)
parser.add_argument('-opt', type=str, default='adam')
parser.add_argument('-out_feature_dim', type=none_or_int, default=11)
parser.add_argument('-pool_strat', type=str, default='SAG')
parser.add_argument('-progress_bar', type=t_or_f, default=False)
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
parser.add_argument('-save_visual', type=t_or_f, default=True)
parser.add_argument('-save_losses', type=t_or_f, default=True)
parser.add_argument('-save_mesh', type=t_or_f, default=True)
parser.add_argument('-save_plot_dir', type=str, default='plots/'+day)
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
    m_ids, m_gs, e_s, args.max_latent_nodes = merge_dataset_stats(train_data, test_data, val_data)
    
    # Save and load m_ids, m_gs, and e_s. Only saves if they don't exist. 
    args.graph_structure_dir = os.path.join(args.graph_structure_dir, f'{args.instance_id}')
    # this attribute is also used in encoder ^
    # if not os.path.isdir(args.graph_structure_dir):
    #     os.mkdir(args.graph_structure_dir)
    #     torch.save(dataset.m_ids, os.path.join(args.graph_structure_dir, 'm_ids.pt'))
    #     torch.save(dataset.m_gs, os.path.join(args.graph_structure_dir, 'm_gs.pt'))
    #     torch.save(dataset.e_s, os.path.join(args.graph_structure_dir, 'e_s.pt'))
    # m_ids, m_gs, e_s = torch.load(os.path.join(args.graph_structure_dir,'m_ids.pt')), torch.load(os.path.join(args.graph_structure_dir,'m_gs.pt')), torch.load(os.path.join(args.graph_structure_dir,'e_s.pt'))

    # args.latent_vec_dim = math.ceil(dataset[0].num_nodes*(args.ae_ratio**args.ae_layers))
    # Initialize Model
    if not args.latent_space:
        logger.warning("Model is not going into latent_space")
    
    model = MultiScaleAutoEncoder(args, m_ids, m_gs, e_s)
    model = model.to(args.device)
    if args.load_model:
        model_path = os.path.join(args.save_model_dir, args.model_file)
        logger.info("Loading model")
        assert os.path.isfile(model_path), f"can't find model file at: {model_path}"
        model.load_state_dict(torch.load(model_path))
        logger.success(f"Multi Scale Autoencoder loaded from {args.model_file}")

    if args.make_gif:
        logger.success('Making a gif <3')
        args.model_file = 'gifff'
        make_gif(model, train_data[:300], args)
        logger.success('Made a gif <3')
        exit()

    # Initialize optimizer and scheduler(?)
    optimizer = build_optimizer(args, model.parameters())

    #dataset = dataset[:250] # The rest of the dataset have little variance

    # Split data into train and test
    # train_data, test_data = train_test_split(dataset, test_size=args.test_ratio)

    # Split training data into train and validation data
    #train_data, val_data = train_test_split(
    #    train_data, test_size=args.val_ratio / (1 - args.test_ratio)
    #)
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
                'latent_dim': [128, 256], 
                'lr' : [1e-4, 1e-5, 1e-6],
                'ae_layers': [3, 4, 5],
                'num_blocks': [1, 2, 3]
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
