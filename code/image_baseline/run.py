#!/usr/bin/env python3

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch
import torchvision
import os
import sys
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from loguru import logger
import numpy as np
import copy
import numpy as np
from datetime import datetime
import random
from VAE import VAE
from train import Trainer

class RandomElastic(object):
  def __init__(self, p = 0.5, alpha = 50., sigma = 5.0, fill = 1.):
    self.p = p
    self.elastic = T.ElasticTransform(alpha = alpha, sigma = sigma, fill = fill)
  
  def __call__(self, x):
    rand = random.uniform(0,1)
    if rand < self.p:
      x = self.elastic(x)
      return x
    else:
      return x

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p = 0.5, device = 'cpu'):
        self.std = std
        self.mean = mean
        self.p = 0.5
        self.device = device
        
    def __call__(self, tensor):
        rand = random.uniform(0, 1)
        if rand < self.p:
          t = torch.randn(tensor.size()).to(self.device)
          return tensor +  t * self.std + self.mean
        else:
          return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def main():
  if torch.backends.mps.is_available():
    device = 'mps'
  elif torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'
  logger.success(f'Device : {device}')

  # ----- SETUP ------
  data_path = 'data/cylinder_flow/'
  id = 1
  root = os.path.join(os.path.join(data_path, f'image_traj/root/'))
  image_folder = lambda id : os.path.join(os.path.join(root, f'traj_{id}'))
  I_FOLDER = image_folder(id)
  if not os.path.isdir(I_FOLDER):
    os.mkdir(I_FOLDER)

  # ----- SETUP ------
  test_ratio = .2
  val_ratio = .2
  batch_size = 16

  transform = T.Compose([T.Resize((128, 512)),
                        #T.Grayscale(),
                        T.ToTensor()])
  dataset = torchvision.datasets.ImageFolder(root, transform = transform)
  train_data, test_data = train_test_split(dataset, test_size=test_ratio, shuffle=True)
  train_data, val_data = train_test_split(train_data,
                                          test_size=val_ratio / (1 - test_ratio), 
                                          shuffle=True)
  train_loader = DataLoader(train_data, 
                            batch_size=batch_size, 
                            num_workers=4, 
                            shuffle=True)
  test_loader = DataLoader(test_data, 
                          batch_size=1, 
                          shuffle = True)
  val_loader = DataLoader(val_data, 
                          batch_size=1, 
                          shuffle=True)
  logger.success(f'Data Loaded\n \
                  Train : {len(train_loader) * batch_size}\n \
                  Test : {len(test_loader)}\n \
                  Val : {len(val_loader)}')

  # Pre Train
  augmentation = T.Compose([AddGaussianNoise(device = device), 
                            T.RandomAffine(
                            degrees=(-1, 1),
                            scale=(1, 1.1),
                            interpolation=T.InterpolationMode.BILINEAR),
                            RandomElastic(p=.3, alpha=5., sigma=2.),
                            T.RandomVerticalFlip(p = .2),
                            #T.RandomHorizontalFlip(p = .2)
                          ])
  timestamp = datetime.now().strftime("%Y_%m_%d-%H.%M")
  z = 32
  start_epoch = 0
  no_epochs = 100
  lr = 1e-4
  eps = 1e-5
  beta = 1e-3
  criterion = 'LMSE'
  args = {'epochs' : (no_epochs - start_epoch), 
          'lr' : {lr},
          'eps' : {eps},
          'beta' : {beta},
          'criterion' : {criterion},
          'batch_size' : {batch_size},
          'z' : {z}}

  with open(f"args/args_{timestamp}.txt", 'w') as f:  
    for key, value in args.items():  
        f.write('%s:%s\n' % (key, value))
    f.close()
  net = VAE(channel_in = 3, z = 64, device = device).to(device)
#  model_path = 'model_chkpoints/model_VAE_2023_11_26-14.51.pt'
  model_path = None

  
  optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=eps)
  if model_path is not None:
     state_dict = torch.load(model_path)
     net.load_state_dict(state_dict['state_dict'])
     optimizer.load_state_dict(state_dict['optimizer_state_dict'])
     logger.success(f'Optimizer and Model loaded')

  trainer = Trainer(beta = beta, device = device, criterion=criterion, time = timestamp)
  best_model, loss = trainer.train(no_epochs = no_epochs,
                net = net, 
                augmentation = augmentation, 
                train_loader = train_loader, 
                val_loader = val_loader, 
                optimizer = optimizer,
                device = device,
                start_epoch = start_epoch,
                progress_bar = False)
  test_loss = trainer.test(net = best_model, test_loader=test_loader)
  logger.success(f'Test Loss: {test_loss}')

if __name__ == "__main__":
    
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
    logger.add(sys.stderr, level='SUCCESS')
    main()
