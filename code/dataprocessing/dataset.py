#!/usr/bin/env python3
import os
import pickle
import re
import torch

from torch_geometric.data import Dataset
from loguru import logger
from dataprocessing.utils.helper_pooling import generate_multi_layer_stride, pool_edge_attr


class MeshDataset(Dataset):
  def __init__(self, args, mode):
    self.data_dir = args.data_dir
    self.layer_num = args.ae_layers
    self.mode = mode
    if mode not in ['train', 'test', 'val']:
       self.mode = 'train'
    # gets data file
    self.data_file = os.path.join(self.data_dir, f'{self.mode}')
    self.mm_dir = os.path.join(self.data_dir, 'mm_files/')
    if not os.path.exists(self.mm_dir):
        os.mkdir(self.mm_dir)
    # directory for storing processed datasets
    self.last_idx = 0
    # number of nodes
    self.n = None
    # For normalization, not implemented atm
    
    self.max_latent_nodes = 0
    self.max_latent_edges = 0
    self.trajectories = set(map(lambda str : re.search('\d+', str).group(), self.processed_file_names))
    self.m_ids = [{} for _ in range(self.layer_num)]
    self.m_gs = [{} for _ in range(self.layer_num + 1)]
    self.e_s = [{} for _ in range(self.layer_num )] 
    self.e_as = {traj : None for traj in self.trajectories}
    self._get_bi_stride()
    super().__init__(self.data_dir)
  
  def _get_bi_stride(self):
    for t in self.trajectories:
      f = next(filter(lambda str : str.startswith(t), self.processed_file_names))
      g = torch.load(os.path.join(self.data_file, f))
      self._cal_multi_mesh(t, g)

  @property
  def processed_file_names(self):
    return os.listdir(self.data_file)

  def len(self):
     return len(self.processed_file_names)  
  
  def get(self, idx):
    file = list(filter(lambda str : str.endswith(f'data_{idx}.pt'), self.processed_file_names))[0]
    return torch.load(os.path.join(self.data_file, file)) # (G, m_ids, m_gs, e_s) -> max m_ids
  
  def _get_pool(self):
     return self.m_ids, self.m_gs, self.e_s, self.e_as

  def __next__(self):
    if self.last_idx == self.len()-1:
      raise StopIteration
    else:
      self.last_idx += 1
      return self.get(self.last_idx)
  
  def _get_latent_attributes(self, graph, m_ids, m_gs):
    edge_attr = graph.edge_attr
    for i in range(self.layer_num):
      mask = m_ids[i]
      new_g = m_gs[i+1]
      edge_attr = pool_edge_attr(mask, new_g, edge_attr, aggr='sum')
    return edge_attr

  def __iter__(self):
    return self

  def _cal_multi_mesh(self, traj, g):
      mmfile = os.path.join(self.mm_dir, str(traj) + '_mmesh_layer_' + str(self.layer_num) + '.dat')
      mmexist = os.path.isfile(mmfile)
      if not mmexist:
          logger.info(f'Calculating multi mesh for trajectory {traj}')
          edge_i = g.edge_index
          n = g.x.shape[0]
          m_gs, m_ids, e_s = generate_multi_layer_stride(edge_i,
                                                    self.layer_num,
                                                    n=n,
                                                    pos_mesh=None)
          m_mesh = {'m_gs': m_gs, 'm_ids': m_ids, 'e_s' : e_s}
          pickle.dump(m_mesh, open(mmfile, 'wb'))
      else:
          logger.info(f'Loaded multi mesh for trajectory {traj}')
          m_mesh = pickle.load(open(mmfile, 'rb'))
          m_gs, m_ids, e_s = m_mesh['m_gs'], m_mesh['m_ids'], m_mesh['e_s']
      if len(m_ids[-1]) > self.max_latent_nodes:
        self.max_latent_nodes = len(m_ids[-1])
      if m_gs[-1].shape[-1] > self.max_latent_edges:
         self.max_latent_edges = m_gs[-1].shape[-1]
      if self.e_as[traj] is None:
         self.e_as[traj] = self._get_latent_attributes(g, m_ids, m_gs)

      for i in range(len(m_ids)):
        self.m_ids[i][str(traj)] = torch.tensor(m_ids[i])
      for j in range(len(m_gs)):
        self.m_gs[j][str(traj)] = m_gs[j]
      for k in range(len(e_s)):
        self.e_s[k][str(traj)] = torch.tensor(e_s[k])


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

  def _cal_multi_mesh(self, traj, g):
      mmfile = os.path.join(self.mm_dir, str(traj) + '_mmesh_layer_' + str(self.layer_num) + '.dat')
      mmexist = os.path.isfile(mmfile)
      if not mmexist:
          edge_i = g.edge_index
          n = g.x.shape[0][0]
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



  