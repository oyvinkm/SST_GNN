#!/bin/python
import random
import numpy as np
from model import MultiScaleAutoEncoder
from opt import build_optimizer
from train import train, test
import torch
from torch_geometric.loader import DataLoader
from loguru import logger
from sklearn.model_selection import train_test_split


from dataprocessing.dataset import MeshDataset

print("CUDA is available:", torch.cuda.is_available())
print("CUDA has version:", torch.version.cuda)


# TODO: Set up args so they can be called from config file
# NOTE: Set args up to take loss function.


def main():
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
    for args in [
        {   
            "datadir" : 'data/cylinder_flow/',
            "instance_id" : 1,
            "normalize" : False,
            "epochs" : 1,
            "test_ratio" : 0.1,
            "val_ratio" : 0.1,
            "batch_size" : 32,
            "shuffle" : True,
            "num_workers" : 1,
            "transforms" : None,
            "num_layers" : 2,
            "out_feature_dim" : 11,
            "ae_layers" : 2,
            "ae_ratio" : 0.5,
            "in_dim_node" : 11,
            "hidden_dim" : 1, ### 64
            "in_dim_edge" : 3,
            "mpl_layers" : 1, ### 2
            "num_blocks" : 1, ### 2
            "latent_dim" : None,
            "pool_strat" : 'ASA',
            "mpl_ratio" : 0.5,
            "opt" : 'adam',
            "lr" : 0.001,
            "weight_decay" : 0.0005,
            "opt_decay_step" : 30,
            "opt_scheduler" : None,
            'save_model' : True,
            'save_model_dir' : 'model_chkpoints',
            'save_args_dir' : 'args_chkpoints'
        },
    ]:
        args = objectview(args)

    # To ensure reproducibility the best we can, here we control the sources of
    # randomness by seeding the various random number generators used in this Colab
    # For more information, see:
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(5)  # Torch
    random.seed(5)  # Python
    np.random.seed(5)  # NumPy

    # Set device to cuda if availale
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize dataset, containing one trajecotry.
    # NOTE: This will be changed to only take <args>
    dataset = MeshDataset(
        root_dir="data/cylinder_flow/",
        instance_id=1,
        layer_num=5)
    args.in_dim_node, args.in_dim_edge = dataset[0].num_features, dataset[0].edge_attr.shape[1]

    # Initialize Model
    model = MultiScaleAutoEncoder(args)

    # Initialize optimizer and scheduler(?)
    scheduler, optimizer = build_optimizer(args, model.parameters())

    # Split data into train and test 
    train_data, test_data = train_test_split(dataset, test_size=args.test_ratio)

    # Split training data into train and validation data
    # TODO: calculate correct val_ratio
    train_data, val_data = train_test_split(train_data, test_size=args.val_ratio)

    #Create Dataloaders for train, test and validation
    # TODO: transform in dataloader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)




    # TRAINING

    train_losses, val_losses, model = train(model = model,  
                                            train_loader = train_loader, 
                                            val_loader = val_loader, 
                                            optimizer = optimizer, 
                                            args = args)

    test_loss = test(model = model,     
                    test_loader = test_loader, 
                    loss_func = None,
                    args = args)

# TESTING
# test()

if __name__ == '__main__':
    main()