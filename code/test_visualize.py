from utils.utils import make_animation
from dataprocessing.preprocessed import loadh5py
from torch_geometric.loader import DataLoader
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

data_list = loadh5py('train', no_trajectories=5)
print(len(data_list))
#loader = DataLoader(data_list)

#make_animation(data_list, 'animation/', 'animation', skip = 30)