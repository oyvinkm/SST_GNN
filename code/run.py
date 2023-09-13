#!/usr/bin/env python3
from dataprocessing.utils.normalization import get_stats
import torch
import numpy as np
import random
from train import train
from utils.utils import save_plots
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

for args in [
        {'model_type': 'meshgraphnet',  
         'num_layers': 10,
         'batch_size': 16, 
         'hidden_dim': 10, 
         'epochs': 500,
         'opt': 'adam', 
         'opt_scheduler': 'none', 
         'opt_restart': 0, 
         'weight_decay': 5e-4, 
         'lr': 0.001,
         'train_size': 45, 
         'test_size': 10, 
         'device':'cuda',
         'shuffle': True, 
         'save_velo_val': True,
         'save_best_model': True, 
         'checkpoint_dir': './best_models/',
         'postprocess_dir': './2d_loss_plots/'},
    ]:
        args = objectview(args)

#To ensure reproducibility the best we can, here we control the sources of
#randomness by seeding the various random number generators used in this Colab
#For more information, see: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(5)  #Torch
random.seed(5)        #Python
np.random.seed(5)     #NumPy

dataset = torch.load('data/trajectories/trajectory_0.pt')[:(args.train_size+args.test_size)]
stats_list = get_stats(dataset)
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
args.device = device
print(device)
test_losses, losses, velo_val_losses, best_model, best_test_loss, test_loader = train(dataset, device, stats_list, args)

print("Min test set loss: {0}".format(min(test_losses)))
print("Minimum loss: {0}".format(min(losses)))
if (args.save_velo_val):
    print("Minimum velocity validation loss: {0}".format(min(velo_val_losses)))
save_plots(args, losses, test_losses, velo_val_losses)