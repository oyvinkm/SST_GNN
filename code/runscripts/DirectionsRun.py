#!/usr/bin/env python3
"""Main file for running setup, training and testing"""
import argparse
import sys
import torch
import os
import json 
from datetime import datetime
import warnings
from torch_geometric.loader import DataLoader
from loguru import logger


sys.path.append('../')
from utils.parserfuncs import none_or_str, none_or_int, none_or_float, t_or_f
from model.model import make_vgae
from dataprocessing.utils.loading import split_pairs
from model.utility import save_run_params, DEFORMATOR_TYPE_DICT
from model.deformator import LatentDeformator
from model.shiftpredictor import ResNetShiftPredictor, LeNetShiftPredictor
from latent_trainer import Params, Trainer, save_difference_norms
from dataprocessing.dataset import MeshDataset

def load_args():
    args = argparse.Namespace()
    f = os.path.join('..','logs','args','SOTAargs.json')
    with open(f) as json_file:
        d = json.load(json_file)
        for key, val in d.items():
            args.__dict__[key] = val 
    return args

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=2000)
parser.add_argument('-logger_lvl', type=str, default='DEBUG')
parser.add_argument('-time_stamp', type=none_or_str, default=datetime.now().strftime("%Y_%m_%d-%H.%M"))
deformator_args = parser.parse_args()

def main():
    vgae_args = load_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logger.remove(0)
    logger.add(sys.stderr, level=deformator_args.logger_lvl)
    vgae_args.device = ("cuda" if torch.cuda.is_available() else "cpu")

    # 'deformator' : 'proj', seems to be best performing
    direction_args = {
        'deformator' : 'proj',
        'directions_count' : 1,
        'out' : 'save_location',
        'gen_path' : os.path.join(vgae_args.save_model_dir, vgae_args.model_file),
        'def_random_init' : True,
        'n_steps' : int(10)
    }

    save_run_params(direction_args)
    

    deformator = LatentDeformator(shift_dim = vgae_args.latent_dim,
                                  input_dim = vgae_args.latent_dim,
                                  out_dim = vgae_args.latent_dim,
                                  type = DEFORMATOR_TYPE_DICT[direction_args['deformator']],
                                  random_init = direction_args['def_random_init']).to(vgae_args.device)
    
    dataset_pairs = torch.load(os.path.join('..','data','latent_space','encoded_dataset_pairs.pt'))
    train_set, validation_set = split_pairs(dataset_pairs)
    save_difference_norms(train_set)
    
    train_loader = DataLoader(
        train_set, batch_size=16, shuffle=vgae_args.shuffle)
    validation_loader = DataLoader(
        validation_set, batch_size=1, shuffle=vgae_args.shuffle)

    params = Params(**direction_args)
    trainer = Trainer(params, out_dir=direction_args['out'], device=vgae_args.device)
    trainer.train(deformator, train_loader, validation_loader, deformator_args)

    # save_results_charts(G, deformator, params, trainer.log_dir, device = vgae_args.device)

if __name__ == '__main__':
    main()