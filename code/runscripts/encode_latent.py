import argparse
#import math
import os
#import random
import sys
#import warnings
from datetime import datetime
#import numpy as np
import torch
from loguru import logger
#from sklearn.model_selection import train_test_split
#from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader
#from sklearn.model_selection import ParameterGrid
#from random import choice
sys.path.append('../')
sys.path.append('dataprocessing')
sys.path.append('model')
sys.path.append('utils')
from dataprocessing.dataset import MeshDataset
from model.model import MultiScaleAutoEncoder
#from model.encoder import Encoder
#from utils.visualization import plot_dual_mesh, make_gif, plot_test_loss, plot_loss
#from utils.opt import build_optimizer
#from train import test, train


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
parser.add_argument('-ae_layers', type=int, default=2)
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-data_dir', type=str, default='../data/cylinder_flow/')
parser.add_argument('-epochs', type=int, default=101)
parser.add_argument('-edge_conv', type=t_or_f, default=True)
parser.add_argument('-hidden_dim', type=int, default=32)
parser.add_argument('-instance_id', type=int, default=1)
parser.add_argument('-latent_space', type=t_or_f, default=True)
parser.add_argument('-logger_lvl', type=str, default='DEBUG')
parser.add_argument('-loss', type=none_or_str, default='LMSE')
parser.add_argument('-load_model', type=t_or_f, default=True)
parser.add_argument('-loss_step', type=int, default=10)
parser.add_argument('-log_step', type=int, default=20)
parser.add_argument('-latent_dim', type=int, default=128)
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-make_gif', type=t_or_f, default=False)
parser.add_argument('-model_file', type=str, default="../logs/model_chkpoints/06-02-24/model_24_02_06-09.56/model.pt")
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
parser.add_argument('-transform', type=t_or_f, default=True)
parser.add_argument('-transform_p', type=float, default=0.1)
parser.add_argument('-time_stamp', type=none_or_str, default=datetime.now().strftime("%Y_%m_%d-%H.%M"))
parser.add_argument('-test_ratio', type=float, default=0.2)
parser.add_argument('-val_ratio', type=float, default=0.1)
parser.add_argument('-weight_decay', type=float, default=0.0005)
args = parser.parse_args()

logger.remove(0)

logger.add(sys.stderr, level=args.logger_lvl)

if torch.cuda.is_available():
    args.device = "cuda"
else:
    args.device = "cpu"
logger.info(f"Device : {args.device}")


dataset = MeshDataset(args=args)
args.in_dim_node, args.in_dim_edge, args.n_nodes = (
  dataset[0].num_features,
  dataset[0].edge_attr.shape[1],
  dataset[0].x.shape[0]
)
model = MultiScaleAutoEncoder(args, dataset.m_ids, dataset.m_gs, dataset.e_s)
model = model.to(args.device)
if args.load_model:
  model_path = args.model_file
  logger.info("Loading model")
  logger.debug(f"{model_path}")
  assert os.path.isfile(model_path), f"can't find model file at: {model_path}"
  model.load_state_dict(torch.load(model_path))
  logger.success(f"Multi Scale Autoencoder loaded from {args.model_file}")
else:
  logger.error(f'Model not loaded')

encoder = model.encoder
pair_list = []
loader = DataLoader(dataset, batch_size = 1)
samp = next(iter(loader))
FOLDER_PATH = f'../data/cylinder_flow/latent_space'
if not os.path.isdir(FOLDER_PATH):
  os.mkdir(FOLDER_PATH)
FILE_PATH = os.path.join(FOLDER_PATH, f'{args.instance_id}_latent_time.pt')
for i, b_data in enumerate(loader):
  _, z, _ = encoder(b_data.to(args.device))
  if i == 0:
    logger.debug(f'{z.shape=} {b_data.t.item()=}')
  if os.path.isfile(FILE_PATH):
    pair_list = torch.load(FILE_PATH)  
  pair_list.append((z, b_data.t.item()))
  torch.save(pair_list, FILE_PATH)
  del pair_list






