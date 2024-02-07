#!/usr/bin/env python3
"""Main file for running setup, training and testing"""
import argparse
import sys
import torch
import os
from datetime import datetime
import warnings
from torch_geometric.loader import DataLoader
from loguru import logger


sys.path.append('../')
from model.model import make_vgae
from dataprocessing.utils.loading import split_pairs
from model.utility import save_run_params, DEFORMATOR_TYPE_DICT
from model.deformator import LatentDeformator
from model.shiftpredictor import ResNetShiftPredictor, LeNetShiftPredictor
from latent_trainer import Params, Trainer


from dataprocessing.dataset import MeshDataset


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
parser.add_argument('-data_dir', type=str, default='../data/cylinder_flow/')
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
parser.add_argument('-model_file', type=str, default="sst_gvae.pt")
parser.add_argument('-mpl_ratio', type=float, default=0.8)
parser.add_argument('-mpl_layers', type=int, default=1)
parser.add_argument('-normalize', type=t_or_f, default=False)
parser.add_argument('-num_blocks', type=int, default=1)
parser.add_argument('-num_workers', type=int, default=1)
parser.add_argument('-n_nodes', type=int, default=1876)
parser.add_argument('-opt', type=str, default='adam')
parser.add_argument('-out_feature_dim', type=none_or_int, default=11)
parser.add_argument('-pool_strat', type=str, default='SAG')
parser.add_argument('-progress_bar', type=t_or_f, default=False)
parser.add_argument('-random_search', type=t_or_f, default=False)
parser.add_argument('-residual', type=t_or_f, default=True)
parser.add_argument('-save_args_dir', type=str, default='../logs/args/'+day)
parser.add_argument('-save_accuracy_dir', type=str, default='../logs/accuracies/'+day)
parser.add_argument('-save_graphstructure_dir', type=str, default='../logs/graph_structure/')
parser.add_argument('-save_gif_dir', type=str, default='../logs/gifs/'+day)
parser.add_argument('-save_loss_over_t_dir', type=str, default='../logs/loss_over_t/'+day)
parser.add_argument('-save_mesh_dir', type=str, default='../logs/meshes/'+day)
parser.add_argument('-save_model_dir', type=str, default='../logs/model_chkpoints/')
parser.add_argument('-save_visualize_dir', type=str, default='../logs/visualizations/'+day)
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
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logger.remove(0)
    logger.add(sys.stderr, level=args.logger_lvl)
    args.device = ("cuda" if torch.cuda.is_available() else "cpu")

    # graph_structure_dir = os.path.join(args.save_graphstructure_dir, f'{args.instance_id}')
    # m_ids, m_gs, e_s = torch.load(os.path.join(graph_structure_dir,'m_ids.pt')), torch.load(os.path.join(graph_structure_dir,'m_gs.pt')), torch.load(os.path.join(graph_structure_dir,'e_s.pt'))

    direction_args = {
        'deformator' : 'proj',
        'directions_count' : 1,
        # 'shift_predictor' : 'LeNet',
        'out' : 'save_location',
        'gen_path' : os.path.join(args.save_model_dir, args.model_file),
        'def_random_init' : True,
        'n_steps' : int(10)
    }

    save_run_params(direction_args)
    
    # G = make_vgae(args, m_ids, m_gs, e_s)

    deformator = LatentDeformator(shift_dim = args.latent_dim,
                                  input_dim = args.latent_dim,
                                  out_dim = args.latent_dim,
                                  type = DEFORMATOR_TYPE_DICT[direction_args['deformator']],
                                  random_init = direction_args['def_random_init']).to(args.device)
    
    dataset_pairs = torch.load(os.path.join('..','data','latent_space','encoded_dataset_pairs.pt'))
    train_set, validation_set = split_pairs(dataset_pairs)
    
    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=args.shuffle)
    validation_loader = DataLoader(
        validation_set, batch_size=1, shuffle=args.shuffle)

    params = Params(**direction_args)
    trainer = Trainer(params, out_dir=direction_args['out'], device=args.device)
    trainer.train(deformator, train_loader, validation_loader, args)

    # save_results_charts(G, deformator, params, trainer.log_dir, device = args.device)

if __name__ == '__main__':
    main()