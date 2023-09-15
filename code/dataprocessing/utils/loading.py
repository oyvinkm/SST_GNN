
import torch
import random
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import os
import numpy as np
import torch
from torch_geometric.data import Data
from .normalization import get_stats
from .triangle_to_edges import triangles_to_edges, NodeType
import h5py
import tensorflow as tf


# OS: 
# 'Linux' : Linux
# 'Darwin' : MacOS
# 'Windows' : 




def load_preprocessed(args):
  """
  Args:
    args.file_path
    train_size: int size of training data
    test_size: int size of test data
    batch_size: int batch size 
    shuffle: bool shuffle dataset

  returns train_loader, test_loader, stats_list
  """
  dataset = torch.load(args.file_path)[:args.train_size + args.test_size]
  if args.shuffle:
    random.shuffle(dataset)
  stats_list = get_stats(dataset)
  train_loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
  test_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
  return train_loader, test_loader, stats_list



def loadh5py(filename, no_trajectories = 1):
  """
  Loads no_trajectories from h5py file
  file : either, test, train or valid
  """
  dataset_dir = 'data/cylinder_flow'
  #Define the data folder and data file name
  if filename not in ['test', 'train', 'valid']:
     filename = 'test'
  datafile = os.path.join(dataset_dir + f'/{filename}.h5')
  data = h5py.File(datafile, 'r')
  #Define the list that will return the data graphs
  data_list = []

  #define the time difference between the graphs
  dt=0.01   #A constant: do not change!

  #define the number of trajectories and time steps within each to process.
  #note that here we only include 2 of each for a toy example.
  number_ts = 600

  with h5py.File(datafile, 'r') as data:

      for i,trajectory in enumerate(data.keys()):
          if(i==no_trajectories):
              break
          print("Trajectory: ",i)

          #We iterate over all the time steps to produce an example graph except
          #for the last one, which does not have a following time step to produce
          #node output values
          for ts in range(len(data[trajectory]['velocity'])-1):
              if(ts==number_ts):
                  break

              #Get node features

              #Note that it's faster to convert to numpy then to torch than to
              #import to torch from h5 format directly
              momentum = torch.tensor(np.array(data[trajectory]['velocity'][ts]))
              #node_type = torch.tensor(np.array(data[trajectory]['node_type'][ts]))
              node_type = torch.tensor(np.array(tf.one_hot(tf.convert_to_tensor(data[trajectory]['node_type'][0]), NodeType.SIZE))).squeeze(1)
              x = torch.cat((momentum,node_type),dim=-1).type(torch.float)

              #Get edge indices in COO format
              edges = triangles_to_edges(tf.convert_to_tensor(np.array(data[trajectory]['cells'][ts])))

              edge_index = torch.cat( (torch.tensor(edges[0].numpy()).unsqueeze(0) ,
                          torch.tensor(edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)

              #Get edge features
              u_i=torch.tensor(np.array(data[trajectory]['pos'][ts]))[edge_index[0]]
              u_j=torch.tensor(np.array(data[trajectory]['pos'][ts]))[edge_index[1]]
              u_ij=u_i-u_j
              u_ij_norm = torch.norm(u_ij,p=2,dim=1,keepdim=True)
              edge_attr = torch.cat((u_ij,u_ij_norm),dim=-1).type(torch.float)

              #Node outputs, for training (velocity)
              v_t=torch.tensor(np.array(data[trajectory]['velocity'][ts]))
              v_tp1=torch.tensor(np.array(data[trajectory]['velocity'][ts+1]))
              y=((v_tp1-v_t)/dt).type(torch.float)

              #Node outputs, for testing integrator (pressure)
              p=torch.tensor(np.array(data[trajectory]['pressure'][ts]))

              #Data needed for visualization code
              cells=torch.tensor(np.array(data[trajectory]['cells'][ts]))
              mesh_pos=torch.tensor(np.array(data[trajectory]['pos'][ts]))

              data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,y=y,p=p,
                                    cells=cells,mesh_pos=mesh_pos))
  return data_list


def save_data_list(data_list, file, data_folder = None):
  """
  data_list : list of graphs loaded from a .h5 file
  file: name of the file you wish to save
  """
  if data_folder is None: 
    data_folder = 'data/trajectories'
  if not os.path.exists(data_folder):
        os.makedirs(data_folder)
  torch.save(data_list, os.path.join(data_folder, f'{file}.pt'))





