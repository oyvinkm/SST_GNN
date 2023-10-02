#!/usr/bin/env python3
import os
import numpy as np
import os.path as osp
from torch_geometric.data import Data, Dataset
import torch
from .utils.helper_pooling import generate_multi_layer_stride
import pickle

class DatasetBase():
  def __init__(self, max_epochs=1, files=None):
    
    self.open_tra_num = 10
    self.file_handle = files
    self.shuffle_file()
    self.data_keys = ("pos", "node_type", "velocity", "cells", "pressure")
    self.out_keys = list(self.data_keys) + ["time"]

    self.tra_idx = 0
    self.epoch_num = 1
    
    # dataset attr
    self.tra_len = 600
    self.time_interval = .01
    self.opened_tra = []
    self.opened_tra_readed_idx = {}
    self.opened_tra_readed_rnd_idx = {}
    self.tra_data = {}
    self.max_epochs = max_epochs

  def open_tra(self):
    while(len(self.opened_tra) < self.open_tra_num):
      tra_idx = self.datasets[self.tra_idx]

      if tra_idx not in self.opened_tra:
        self.opened_tra.append(tra_idx)
        self.opened_tra_readed_idx[tra_idx] = -1
        self.opened_tra_readed_rnd_idx[tra_idx] = np.random.permutation(self.tra_len - 2)
      self.tra_idx += 1

      if self.check_if_epoch_end():
        self.epoch_end()
        print('Epoch Finished')

  def check_and_close_tra(self):
    to_del = []
    for tra in self.opened_tra:
      if self.opened_tra_readed_index[tra] >= (self.tra_len - 3):
        to_del.append(tra)
      for tra in to_del:
        self.opened_tra.remove(tra)
        try:
          del self.opened_tra_readed_idx[tra]
          del self.opened_tra_readed_rnd_idx[tra]
          del self.tra_data[tra]
        except Exception as e:
          print(e)

  def shuffle_file(self):
    datasets = list(self.file_handle.keys())
    np.random.shuffle(datasets)
    self.datasets = datasets
  
  def epoch_end(self):
    self.tra_idx = 0
    self.shuffle_file()
    self.epoch_num += 1

  def check_if_epoch_end(self):
    if self.tra_idx >= len(self.file_handle):
      return True
    return False
  
  @staticmethod
  def datas_to_graph(datas):
    time_vector = np.ones((datas[0].shape[0], 1))*datas[5]
    node_attr = np.hstack((datas[1], datas[2][0], datas[4][0], time_vector))
    
    "node_type, cur_v, pressure, time"
    crds = torch.as_tensor(datas[0], dtype=torch.float)

    target = datas[2][1]
    node_attr = torch.as_tensor(node_attr, dtype=torch.float)
    target = torch.from_numpy(target)
    face = torch.as_tensor(datas[3].T, dtype=torch.long)
    g = Data(x=node_attr, face=face, y=target, pos=crds)
    return g

  def __next__(self):
    self.check_and_close_tra()
    self.open_tra()

    if self.epoch_num > self.max_epochs:
      raise StopIteration
    
    selected_tra = np.random.choice(self.opened_tra)

    data = self.tra_data.get(selected_tra, None)
    if data is None:
      data = self.file_handle[selected_tra]
      self.tra_data[selected_tra] = data
    
    selected_tra_readed_idx = self.opened_tra_readed_idx[selected_tra]
    selected_frame = self.opened_tra_readed_rnd_idx[selected_tra][selected_tra_readed_idx+1]
    self.opened_tra_readed_idx[selected_tra] += 1
    datas = []
    for k in self.data_keys:
      if k in ['velocity', 'pressure']:
        r = np.array((data[k][selected_frame], data[k][selected_frame + 1]), dtype=np.float32)
      else:
        r = data[k][selected_frame]
        if k in ['node_type', 'cells']:
          r = r.astype(np.int32)
      datas.append(r)
    datas.append(np.array([self.time_interval * selected_frame], dtype=np.float32))
    #('pos', 'node_type', 'velocity', 'cells', 'pressure', 'time')
    g = self.datas_to_graph(datas)
    return g
  
class MeshDataset(Dataset):
  def __init__(self,
                 root_dir,
                 instance_id,
                 layer_num,
                 normalize = True,
                 recal_mesh = True):
    self.root_dir = root_dir
    self.instance_id = instance_id
    self.layer_num = layer_num
    self.normalize = normalize
    self.recal_mesn = recal_mesh
    self.data_file = os.path.join(self.root_dir, f'trajectories/trajectory_{str(instance_id)}.pt')
    self.mm_dir = os.path.join(self.root_dir, 'mm_files/')
    if not os.path.exists(self.mm_dir):
        os.mkdir(self.mm_dir)
    try:
      self.traj_data = torch.load(self.data_file)
    except:
      FileExistsError, f'{self.data_file} does not exist'

    self._cal_multi_mesh()

  def len(self):
     return len(self.traj_data)  
  
  def get(self, idx):
     return self.traj_data[idx]
     

  def _cal_multi_mesh(self):
      mmfile = os.path.join(self.mm_dir, str(self.instance_id) + '_mmesh_layer_' + str(self.layer_num) + '.dat')
      mmexist = os.path.isfile(mmfile)
      if not mmexist:
          edge_i = self.traj_data[0].edge_index
          n = self.traj_data[0].x.shape[0]
          m_gs, m_ids = generate_multi_layer_stride(edge_i,
                                                    self.layer_num,
                                                    n=n,
                                                    pos_mesh=None)
          m_mesh = {'m_gs': m_gs, 'm_ids': m_ids}
          pickle.dump(m_mesh, open(mmfile, 'wb'))
      else:
          m_mesh = pickle.load(open(mmfile, 'rb'))
          m_gs, m_ids = m_mesh['m_gs'], m_mesh['m_ids']
      self.m_ids = m_ids
      self.m_gs = m_gs







  