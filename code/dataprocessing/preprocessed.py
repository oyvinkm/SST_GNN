#!/usr/bin/env python3
import torch
import random
import pandas as pd
import torch_scatter
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
from utils.normalization import normalize, unnormalize, get_stats
import platform


# OS: 
# 'Linux' : Linux
# 'Darwin' : MacOS
# 'Windows' : 



def filepath(filename = None):
  root_dir = os.path.pardir
  dataset_dir = os.path.join(root_dir, 'data/preprocessed/')
  if filename == '30':
    file_path = os.path.join(dataset_dir, 'meshgraphnets_miniset30traj5ts_vis.pt')
  elif filename == '100':
    file_path = os.path.join(dataset_dir, 'meshgraphnets_miniset100traj25ts_vis.pt')
  elif filename == 'test':
    file_path = os.path.join(dataset_dir, 'test_processed_set.pt')
  else: 
    file_path = os.path.join(dataset_dir, 'meshgraphnets_miniset5traj_vis.pt')
  return file_path

def loadData(args):
  """
  args: objectview with arguments for training.
  dataset: which dataset to load, length = train_size + test_size
  train_size: int: size of training data
  test_size: int: size of test data
  batch_size: int: batch size 
  shuffle: bool: shuffle dataset

  returns train_loader, test_loader, stats_list
  """
  file_path = filepath(args.dataset)
  dataset = torch.load(file_path)[:args.train_size + args.test_size]
  if args.shuffle:
    random.shuffle(dataset)
  stats_list = get_stats(dataset)
  train_loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
  test_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)
  return train_loader, test_loader, stats_list



# loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
# test_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)


