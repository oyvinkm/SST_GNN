#!/usr/bin/env python3
import os
import pickle
from typing import Callable, Optional

from dataprocessing.utils.helper_pooling import generate_multi_layer_stride
from dataprocessing.utils.normalization import get_stats
from loguru import logger
import torch
from torch_geometric.data import Dataset


class MeshDataset(Dataset):
  def __init__(self, args):
    self.data_dir = args.data_dir
    self.instance_id = args.instance_id
    self.normalize = args.normalize
    self.layer_num = args.ae_layers
    # gets data file
    self.data_file = os.path.join(self.data_dir, f'trajectories/trajectory_{str(self.instance_id)}.pt')
    self.mm_dir = os.path.join(self.data_dir, 'mm_files/')
    if not os.path.exists(self.mm_dir):
        os.mkdir(self.mm_dir)
    # directory for storing processed datasets
    #self.mm_dir = os.path.join(self.data_dir, 'mm_files/')
    self.last_idx = 0
    # number of nodes
    self.n = None
    
    self.traj_data = torch.load(self.data_file)
    
    # For normalization, not implemented atm
    self.stats_list = get_stats(self.traj_data)
    self._cal_multi_mesh()
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

  def _cal_multi_mesh(self):
      mmfile = os.path.join(self.mm_dir, str(self.instance_id) + '_mmesh_layer_' + str(self.layer_num) + '.dat')
      mmexist = os.path.isfile(mmfile)
      if not mmexist:
          edge_i = self.traj_data[0].edge_index
          n = self.traj_data[0].x.shape[0]
          logger.debug(f'n : {n}')
          m_gs, m_ids, e_s = generate_multi_layer_stride(edge_i,
                                                    self.layer_num,
                                                    n=n,
                                                    pos_mesh=None)
          m_mesh = {'m_gs': m_gs, 'm_ids': m_ids, 'e_s' : e_s}
          pickle.dump(m_mesh, open(mmfile, 'wb'))
      else:
          m_mesh = pickle.load(open(mmfile, 'rb'))
          m_gs, m_ids, e_s = m_mesh['m_gs'], m_mesh['m_ids'], m_mesh['e_s']
      self.m_ids = m_ids
      self.m_gs = m_gs
      self.e_s = e_s


class DatasetPairs(Dataset):
  def __init__(self, args):
    self.data_dir = args.data_dir
    self.instance_id = args.instance_id
    self.normalize = args.normalize
    self.layer_num = args.ae_layers
    # gets data file

    if args.train:
      self.data_file = os.path.join(self.data_dir, f'pairs/train_pair_{str(self.instance_id)}.pt')
    else:
      self.data_file = os.path.join(self.data_dir, f'pairs/test_pair_{str(self.instance_id)}.pt')
    self.mm_dir = os.path.join(self.data_dir, 'mm_files/')
    if not os.path.exists(self.mm_dir):
        os.mkdir(self.mm_dir)
    # directory for storing processed datasets
    #self.mm_dir = os.path.join(self.data_dir, 'mm_files/')
    self.last_idx = 0
    # number of nodes
    self.n = None
    
    self.traj_data = torch.load(self.data_file)
    self._cal_multi_mesh()
    super().__init__(self.data_dir)
  
  def len(self):
     return len(self.traj_data)  
  
  def get(self, idx):
    input, label = self.traj_data[idx]
    return (input, label)
    

  def __next__(self):
    if self.last_idx == self.len()-1:
      raise StopIteration
    else:
      self.last_idx += 1
      return self.get(self.last_idx)

  def __iter__(self):
    return self

  def _cal_multi_mesh(self):
      mmfile = os.path.join(self.mm_dir, str(self.instance_id) + '_mmesh_layer_' + str(self.layer_num) + '.dat')
      mmexist = os.path.isfile(mmfile)
      if not mmexist:
          edge_i = self.traj_data[0][0].edge_index
          n = self.traj_data[0][0].x.shape[0][0]
          m_gs, m_ids, e_s = generate_multi_layer_stride(edge_i,
                                                    self.layer_num,
                                                    n=n,
                                                    pos_mesh=None)
          m_mesh = {'m_gs': m_gs, 'm_ids': m_ids, 'e_s' : e_s}
          pickle.dump(m_mesh, open(mmfile, 'wb'))
      else:
          m_mesh = pickle.load(open(mmfile, 'rb'))
          m_gs, m_ids, e_s = m_mesh['m_gs'], m_mesh['m_ids'], m_mesh['e_s']
      self.m_ids = m_ids
      self.m_gs = m_gs
      self.e_s = e_s



  