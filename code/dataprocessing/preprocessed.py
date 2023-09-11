#!/usr/bin/env python3
import torch
import random
import pandas as pd
# import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import DataLoader
import numpy as np
import torch.optim as optim
from tqdm import trange
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch_geometric.data import Data
from utils.normalization import *

"""
Preprocessed data """
dataset_dir = os.path.join(os.path.pardir, 'data/preprocessed/')
print(dataset_dir)
file_path=os.path.join(dataset_dir, 'meshgraphnets_miniset5traj_vis.pt')
print(file_path)
dataset_full_timesteps = torch.load(file_path)
dataset = torch.load(file_path)[:1]
data2 = torch.load(file_path)[1:2]
print(len(dataset_full_timesteps)/5)






# loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
# test_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)


