from utils.visualization import make_animation
from model import MultiScaleAutoEncoder
from torch_geometric.loader import DataLoader
from dataprocessing.dataset import MeshDataset
import argparse
from datetime import datetime
from torch_geometric.data import Batch, Data
import torch
import os
import warnings
import sys
from loguru import logger
import copy

day = datetime.now().strftime("%d-%m-%y")
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
parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', type=str, default='data/cylinder_flow/')
parser.add_argument('-save_args_dir', type=str, default='args')
parser.add_argument('-save_accuracy_dir', type=str, default='accuracies/'+day)
parser.add_argument('-save_visualize_dir', type=str, default='visualizations')
parser.add_argument('-save_mesh_dir', type=str, default='meshes')
parser.add_argument('-ae_pool_strat', type=str, default='SAG')
parser.add_argument('-pool_strat', type=str, default='ASA')
parser.add_argument('-opt', type=str, default='adam')
parser.add_argument('-opt_scheduler', type=str, default='step')
parser.add_argument('-save_model_dir', type=str, default='model')
parser.add_argument('-save_plot_dir', type=str, default='plots')
parser.add_argument('-logger_lvl', type=str, default='DEBUG')
parser.add_argument('-residual_idx', type=list, default = [1])
parser.add_argument('-transform', type=t_or_f, default=False)
parser.add_argument('-time_stamp', type=none_or_str, default=datetime.now().strftime("%Y_%m_%d-%H.%M"))
parser.add_argument('-normalize', type=t_or_f, default=False)
parser.add_argument('-shuffle', type=t_or_f, default=True)
parser.add_argument('-save_plot', type=t_or_f, default=True)
parser.add_argument('-save_model', type=t_or_f, default=True)
parser.add_argument('-save_visual', type=t_or_f, default=True)
parser.add_argument('-save_losses', type=t_or_f, default=True)
parser.add_argument('-save_mesh', type=t_or_f, default=True)
parser.add_argument('-load_model', type=t_or_f, default=True)
parser.add_argument('-model_file', type=str, default='')
parser.add_argument('-loss_step', type=int, default=10)
parser.add_argument('-log_step', type=int, default=10)
parser.add_argument('-test_ratio', type=float, default=0.2)
parser.add_argument('-val_ratio', type=float, default=0.1)
parser.add_argument('-mpl_ratio', type=float, default=0.5)
parser.add_argument('-lr', type=float, default=1e-5)
parser.add_argument('-loss', type=none_or_str, default=None)
parser.add_argument('-weight_decay', type=float, default=0.0005)
parser.add_argument('-opt_decay_rate', type=float, default=0.1)
parser.add_argument('-transform_p', type=float, default=0.1)
parser.add_argument('-ae_ratio', type=none_or_float, default=0.1)
parser.add_argument('-instance_id', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-residual', type=t_or_f, default=True)
parser.add_argument('-epochs', type=int, default=2)
parser.add_argument('-ae_layers', type=int, default=2)
parser.add_argument('-hidden_dim', type=int, default=32)
parser.add_argument('-mpl_layers', type=int, default=2)
parser.add_argument('-num_blocks', type=int, default=1)
parser.add_argument('-opt_decay_step', type=int, default=30)
parser.add_argument('-opt_restart', type=int, default=10)
parser.add_argument('-num_workers', type=int, default=1)
parser.add_argument('-out_feature_dim', type=none_or_int, default=11)
parser.add_argument('-latent_dim', type=int, default=256)
parser.add_argument('-progress_bar', type=t_or_f, default=False)
parser.add_argument('-n_nodes', type=int, default=1876)
args = parser.parse_args()
logger.remove(0)
logger.add(sys.stderr, level=args.logger_lvl)
warnings.filterwarnings(
    "ignore",
    ".*Sparse CSR tensor support is in beta state.*")
args.device = "cpu"
args.model_file = "2023_11_25-16.38_ae_layers-2_ae_ratio-0.3_hidden_dim-64_latent_dim-256_pool_strat-SAG.pt"
args.save_model_dir = "model/25-11-23"
dataset = MeshDataset(args = args)
args.in_dim_node, args.in_dim_edge = (
        dataset[0].num_features,
        dataset[0].edge_attr.shape[1],
    )

model = MultiScaleAutoEncoder(args, dataset.m_ids, dataset.m_gs)
    
model_path = os.path.join(args.save_model_dir , args.model_file)
if args.load_model and os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
    logger.success(f"Multi Scale Autoencoder loaded from {args.model_file}")

PRED = copy.deepcopy(dataset)
GT = copy.deepcopy(dataset)
DIFF = copy.deepcopy(dataset)
for pred_data, gt_data, diff_data in zip(PRED, GT, DIFF):
    with torch.no_grad():
        pred, _ = model(Batch.from_data_list([pred_data]))
        pred_data.x = pred.x
        diff_data.x = pred_data.x - gt_data.x

logger.info("processing done...")

file_dir = "gifs/"
gif_name = args.model_file + "anim.gif"

make_animation(GT, PRED, DIFF, file_dir, gif_name, skip = 50)

logger.success("animation complete...")
