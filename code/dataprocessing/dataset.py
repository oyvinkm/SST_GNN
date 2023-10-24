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
    try:
      self.traj_data = torch.load(self.data_file)
    except:
      FileExistsError, f'{self.data_file} does not exist'
    # For normalization, not implemented atm
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = get_stats(self.traj_data)
    self.mean_vec_x = mean_vec_x
    self.std_vec_x = std_vec_x
    self.mean_vec_edge = mean_vec_edge
    self.std_vec_edge = std_vec_edge
    self.mean_vec_y = mean_vec_y
    self.std_vec_y = std_vec_y
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

  """ def _cal_multi_mesh(self):
      mmfile = os.path.join(self.mm_dir, str(self.instance_id) + '_mmesh_layer_' + str(self.layer_num) + '.dat')
      mmexist = os.path.isfile(mmfile)
      self.n = self.traj_data[0].x.shape[0]
      if not mmexist:
          edge_i = self.traj_data[0].edge_index
          m_gs, m_ids = generate_multi_layer_stride(edge_i,
                                                    self.layer_num,
                                                    n=self.n,
                                                    pos_mesh=None)
          m_mesh = {'m_gs': m_gs, 'm_ids': m_ids}
          pickle.dump(m_mesh, open(mmfile, 'wb'))
      else:
          m_mesh = pickle.load(open(mmfile, 'rb'))
          m_gs, m_ids = m_mesh['m_gs'], m_mesh['m_ids']
      # pooling node indices
      self.m_ids = m_ids
      # pooling edges
      self.m_gs = m_gs
 """






  