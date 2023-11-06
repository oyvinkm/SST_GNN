#!/usr/bin/env python3
import os
from torch_geometric.data import Dataset
import torch
from dataprocessing.utils.normalization import get_stats


class MeshDataset(Dataset):
  def __init__(self, args):
    self.data_dir = args.data_dir
    self.instance_id = args.instance_id
    self.normalize = args.normalize
    # gets data file
    self.data_file = os.path.join(self.data_dir, f'trajectories/trajectory_{str(self.instance_id)}.pt')
    # directory for storing processed datasets
    #self.mm_dir = os.path.join(self.data_dir, 'mm_files/')
    self.last_idx = 0
    # number of nodes
    self.n = None
    #if not os.path.exists(self.mm_dir):
    #    os.mkdir(self.mm_dir)
    self.traj_data = torch.load(self.data_file)
    
    # For normalization, not implemented atm
    self.stats_list = get_stats(self.traj_data)
    #self._cal_multi_mesh()
    super().__init__(self.data_dir)

  def len(self):
     return len(self.traj_data)  
  
  def get(self, idx):
    return self.traj_data[idx]
    

  def __next__(self):
    if self.last_idx == self.len()-1:
      raise StopIteration
    else:
      self.last_idx += 1
      return self.get(self.last_idx)

  def __iter__(self):
    return self





  