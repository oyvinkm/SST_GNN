
import torch
import random
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import os
import json
import numpy as np
import torch

import random
from typing import Optional, Union
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

def find_trajectory_nodes(path = 'data/cylinder_flow/', 
                          mode = 'train', 
                          save_json = True, 
                          json_file = 'same_nodes.json'):
  """
  This function finds the trajectories that share the same amount of nodes in a given dataset 
  and optionally saves the result as a JSON file.
  
  Args:
    path (str, optional): The path to the dataset. Defaults to 'data/cylinder_flow/'.
    mode (str, optional): The .h5 file to read from, can be 'train', 'test', or 'val'. Defaults to 'train'.
    save_json (bool, optional): Whether to save the result as a JSON file. Defaults to True.
    json_file (str, optional): The name of the JSON file to save. Defaults to 'same_nodes.json'.
  
  Returns:
    dict: A dictionary where the keys are the number of nodes in a trajectory and the values are 
          lists of trajectories with that number of nodes.
  """

  if mode not in ['train', 'test', 'val']:
    mode = 'train'
  h5File = os.path.join(path, f'{mode}.h5')
  trajectories_same = {}
  with h5py.File(h5File, 'r') as data:
    for i, trajectory in enumerate(data.keys()):
      num_nodes = data[trajectory]['velocity'].shape[1]
      if num_nodes not in trajectories_same:
        trajectories_same[num_nodes] = []
      trajectories_same[num_nodes].append(trajectory)
  if save_json:
     with open(json_file, 'w') as file:
        json.dump(trajectories_same, file)
  return trajectories_same

def save_trajectory(save_path, trajectories):
   """
   Saves a list of trajectories to a specified directory.
  
   :param save_path: The directory where the trajectories will be saved.
   :type save_path: str
   :param trajectories: The list of trajectories to be saved.
   :type trajectories: list
   """
   if not os.path.isdir(save_path):
      print('saving in folder', save_path)
      os.mkdir(save_path)
   for i,g in enumerate(trajectories):
      torch.save(g, os.path.join(save_path, f'{g.trajectory}_data_{i}.pt'))

def constructDatasetFolders(same_nodes : Optional[Union[str, dict]], 
                            choose : Optional[Union[str, int, None]], 
                            data_dir = 'data/cylinder_flow/', 
                            mode = 'train'):
  """
  Constructs dataset folders for training, validation, and testing.

  Args:
    same_nodes (str or dict, optional): A JSON file path or a dictionary containing nodes data.
    choose (str or int or None, optional): Determines the selection of trajectories. If 'min', the node with the least trajectories is chosen. If an integer, a node with that specific amount of trajectories is chosen. If None, the node with the most trajectories is chosen.
    data_dir (str, optional): The directory where the data is stored. Defaults to 'data/cylinder_flow/'.
    mode (str, optional): The mode of operation. Can be 'train', 'test', or 'val'. Defaults to 'train'.

  Returns:
    None. The function works by side effect, creating directories and saving trajectories in the specified data directory.
  """
  if mode not in ['train', 'test', 'val']:
    mode = 'train'
  if isinstance(same_nodes, str) and same_nodes.endswith('.json'):
    f = open(same_nodes, 'r')
    node_dict = json.loads(f.read())
  else:
    node_dict = same_nodes
  max_len = max(map(len, node_dict.values()))

  if choose == 'min':
     # Choose the one with the least trajectories
     node_key = min(node_dict, key= lambda x: len(set(node_dict[x])))
  elif isinstance(choose, int):
     # Choose one with a specific amount of trajecotries
     choose = max(choose, 0); choose = min(choose, max_len)
     node_key = random.choice(list({k: v for k, v in node_dict.items() if len(v) == 8}.keys()))
  else:
     # Default to choosing the maximum
     node_key = max(node_dict, key= lambda x: len(set(node_dict[x])))
  trajectories = node_dict[node_key]
  traj_dir = os.path.join(data_dir, f'trajectories_{node_key}')
  if not os.path.isdir(traj_dir):
     os.mkdir(traj_dir)
  train = load_trajectories(mode, trajectories[:-2], save = False)
  save_trajectory(os.path.join(traj_dir, 'train'), train)
  print('training set saved')
  val = load_trajectories(mode, [trajectories[-2]], save = False)
  save_trajectory(os.path.join(traj_dir, 'val'), val)
  print('validation set saved')
  test = load_trajectories(mode, [trajectories[-1]], save = False)
  save_trajectory(os.path.join(traj_dir, 'test'), test)
  print('test set saved')
  
  

def load_trajectories(filename, trajectories, save = False, save_folder = None):
  """
  This function loads the trajectories from a given file, processes them and optionally saves them in a specified folder.
  
  Args:
    filename (string): The name of the file to load the trajectories from. It can be either 'test', 'train' or 'valid'.
    trajectories (list): A list of trajectories to process.
    save (boolean): A flag indicating whether to save the processed trajectories or not. Default is False.
    save_folder (string): The name of the folder where to save the processed trajectories. It is used only if save is True.
  
  Returns:
    list: A list of processed trajectories.
  """

  dataset_dir = os.path.join(os.getcwd(), 'data/cylinder_flow')
  if filename not in ['test', 'train', 'valid']:
     filename = 'test'
  datafile = os.path.join(dataset_dir + f'/{filename}.h5')
  #Define the list that will return the data graphs
  data_list = []

  #define the time difference between the graphs
  dt=0.01   #A constant: do not change!

  #define the number of trajectories and time steps within each to process.
  #note that here we only include 2 of each for a toy example.
  number_ts = 600
  with h5py.File(datafile, 'r') as data:
    
    for trajectory in trajectories:
        print("Trajectory: ",trajectory)

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
            tmp = tf.convert_to_tensor(data[trajectory]['node_type'][0])
            node_type = torch.tensor(np.array(tf.one_hot(tf.convert_to_tensor(data[trajectory]['node_type'][0]), NodeType.SIZE))).squeeze(1)
            x = torch.cat((momentum,node_type),dim=-1).type(torch.float)
            if ts == 0: print(f'Num nodes trajectory {trajectory} : {x.shape[0]}')

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
            w = x.new_ones(x.shape[0], 1)
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,y=y,p=p,
                                  cells=cells, weights = w, mesh_pos=mesh_pos, t=ts, trajectory = trajectory))
        if save:
          file = f'trajectory_{trajectory}'
          save_data_list(data_list, file, save_folder)
          data_list = []
  return data_list


def loadh5py(filename, no_trajectories = 1, save = False, save_folder = None):
  """
  Loads no_trajectories from h5py file
  file : either, test, train or valid
  """
  dataset_dir = os.path.join(os.getcwd(), 'data/cylinder_flow')
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
          if(i == no_trajectories):
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
              tmp = tf.convert_to_tensor(data[trajectory]['node_type'][0])
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
              w = x.new_ones(x.shape[0], 1)
              data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,y=y,p=p,
                                    cells=cells, weights = w, mesh_pos=mesh_pos, t=ts))
          if save:
            file = f'trajectory_{trajectory}'
            save_data_list(data_list, file, save_folder)
            data_list = []
  return data_list


def storeh5py(filename, trajectory = 1):
  """
  Loads no_trajectories from h5py file
  file : either, test, train or valid
  """
  dataset_dir = os.path.join(os.getcwd(), 'data/cylinder_flow')
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
      
    
    #We iterate over all the time steps to produce an example graph
    length = len(data[trajectory]['velocity'])
    h5_data = {'x' : np.ndarray()}
    for ts in range(len(data[trajectory]['velocity'])):
        
        #Get node features
        #Note that it's faster to convert to numpy then to torch than to
        #import to torch from h5 format directly
        momentum = torch.tensor(np.array(data[trajectory]['velocity'][ts]))
            
        #node_type = torch.tensor(np.array(data[trajectory]['node_type'][ts]))
        tmp = tf.convert_to_tensor(data[trajectory]['node_type'][0])
        node_type = torch.tensor(np.array(tf.one_hot(tf.convert_to_tensor(data[trajectory]['node_type'][0]), NodeType.SIZE))).squeeze(1)
        x = torch.cat((momentum,node_type),dim=-1).type(torch.float)
        h5_data['x'] = x
        #Get edge indices in COO format
        edges = triangles_to_edges(tf.convert_to_tensor(np.array(data[trajectory]['cells'][ts])))

        edge_index = torch.cat( (torch.tensor(edges[0].numpy()).unsqueeze(0) ,
                    torch.tensor(edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)
        h5_data['edge_index'] = edge_index
        #Get edge features
        u_i=torch.tensor(np.array(data[trajectory]['pos'][ts]))[edge_index[0]]
        u_j=torch.tensor(np.array(data[trajectory]['pos'][ts]))[edge_index[1]]
        u_ij=u_i-u_j
        u_ij_norm = torch.norm(u_ij,p=2,dim=1,keepdim=True)
        edge_attr = torch.cat((u_ij,u_ij_norm),dim=-1).type(torch.float)
        h5_data['edge_attr'] = edge_attr
        #Node outputs, for training (velocity)
        v_t=torch.tensor(np.array(data[trajectory]['velocity'][ts]))
        v_tp1=torch.tensor(np.array(data[trajectory]['velocity'][ts+1]))
        y=((v_tp1-v_t)/dt).type(torch.float)
        h5_data['y'] = y
        #Node outputs, for testing integrator (pressure)
        p=torch.tensor(np.array(data[trajectory]['pressure'][ts]))
        h5_data['p'] = p
        #Data needed for visualization code
        cells=torch.tensor(np.array(data[trajectory]['cells'][ts]))
        h5_data['cells'] = cells
        mesh_pos=torch.tensor(np.array(data[trajectory]['pos'][ts]))
        h5_data['mesh_pos'] = mesh_pos

def save_data_list(data_list, file, data_folder = None):
  """
  data_list : list of graphs loaded from a .h5 file
  file: name of the file you wish to save
  """
  if data_folder is None: 
    data_folder = 'data/cylinder_flow/trajectories'
  if not os.path.exists(data_folder):
        os.makedirs(data_folder)
  torch.save(data_list, os.path.join(data_folder, f'{file}.pt'))


def split_pairs(data, ratio = .1):
  """Splits list of data pairs into two lists and discards overlapping entries"""
  test_data = []
  rng = np.random.default_rng()
  choice = rng.choice(len(data)-1, int(ratio * len(data)), replace = False)
  for i in choice: test_data.append(data[i]); data[i] = None
  for j in choice: data[min(j+1, len(data))] = None; data[max(j-1, 0)] = None
  train_data = list(filter(lambda x : x is not None, data))
  return train_data, test_data

def save_traj_pairs(instance_id, ratio=.1):
  pairs = '../data/cylinder_flow/pairs'
  if not os.path.isdir(pairs):
    os.mkdir(pairs)
  trajectory = lambda id : f'../data/cylinder_flow/trajectories/trajectory_{id}.pt'
  data_list = torch.load(trajectory(instance_id))
  data_list = sorted(data_list, key = lambda g : g.t)
  data_pairs = list(zip(data_list[:-2], data_list[1:-1]))
  train_data, test_data = split_pairs(data_pairs)
  train_path = os.path.join(pairs, f'train_pair_{instance_id}.pt')
  test_path = os.path.join(pairs, f'test_pair_{instance_id}.pt')
  torch.save(train_data, train_path)
  torch.save(test_data, test_path)





