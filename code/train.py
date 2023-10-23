import torch
from tqdm import trange
import copy
from torch.nn import MSELoss
from mask import AttributeMask, EdgeMask
from dataprocessing.utils.normalization import get_stats, normalize, unnormalize
from torch_geometric.data import Batch
import numpy as np
import os
from datetime import datetime
from loguru import logger
import json


# datetime object containing current date and time

def save_model(best_model, model_name, args):
    if not os.path.isdir( args.save_model_dir):
        os.mkdir(args.save_model_dir)
    PATH = os.path.join(args.save_model_dir, model_name+'.pt')
    torch.save(best_model.state_dict(), PATH)

def save_args(args_name, args):
    if not os.path.isdir(args.save_args_dir):
        os.mkdir(args.save_args_dir)
    PATH = os.path.join(args.save_args_dir, args_name+'.txt')
    with open(PATH, 'w') as f:
         for key, value in args.__dict__.items():  
            f.write('%s: %s\n' % (key, value))
@torch.no_grad()
def validate(model, val_loader, loss_func, args):
    total_loss = 0
    model.train()
    for idx, batch in enumerate(val_loader):
        # data = transform(batch).to(args.device)
        #Note that normalization must be done before it's called. The unnormalized
        #data needs to be preserved in order to correctly calculate the loss
        b_lst = batch.to_data_list()
        tmp = []
        for b in b_lst:
            w = b.x.new_ones(b.x.shape[0], 1)
            b.weights = w
            tmp.append(b)
        batch = Batch.from_data_list(tmp)
        batch=batch.to(args.device)
        pred, z = model(batch)
        loss = loss_func(pred.x, batch.x)
        total_loss += loss.item()

    total_loss /= idx
    return total_loss


@torch.no_grad()
def test(model, test_loader, loss_func, args):
    if loss_func is None:
        loss_func = MSELoss()
    total_loss = 0
    model.train()
    for idx, batch in enumerate(test_loader):
        # data = transform(batch).to(args.device)
        #Note that normalization must be done before it's called. The unnormalized
        #data needs to be preserved in order to correctly calculate the loss
        b_lst = batch.to_data_list()
        tmp = []
        for b in b_lst:
            w = b.x.new_ones(b.x.shape[0], 1)
            b.weights = w
            tmp.append(b)
        batch = Batch.from_data_list(tmp)
        batch=batch.to(args.device)
        pred, z = model(batch)
        loss = loss_func(pred.x, batch.x)
        total_loss += loss.item()

    total_loss /= idx
    return total_loss

def train(model, train_loader, val_loader, optimizer, args):

    model = model.to(args.device)
    '''
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    '''
    
    #Define the model name for saving
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y_%H.%M.%S")
    model_name= "model_" + dt_string
    args_name = "args_" + dt_string
    # TODO: Save args in a file with date and time
    

    # train
    # NOTE: Might make dependent on args which loss function
    loss_func = MSELoss()
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        for idx, batch in enumerate(train_loader):
            # data = transform(batch).to(args.device)
            #Note that normalization must be done before it's called. The unnormalized
            #data needs to be preserved in order to correctly calculate the loss
            # NOTE: This is only temrporary
            b_lst = batch.to_data_list()
            tmp = []
            for b in b_lst:
                w = b.x.new_ones(b.x.shape[0], 1)
                b.weights = w
                tmp.append(b)
            batch = Batch.from_data_list(tmp)
            # NOTE: REMEMBER TO REMOVE THHIS
            batch=batch.to(args.device)
            optimizer.zero_grad()         #zero gradients each time
            pred, z = model(batch)
            # NOTE: Does the loss have to be a function in the model? 
            loss = loss_func(pred.x, batch.x)
            loss.backward()         #backpropagate loss
            optimizer.step()
            total_loss += loss.item()

        total_loss /= idx
        train_losses.append(total_loss)

        #Every tenth epoch, calculate acceleration test loss and velocity validation loss
        if epoch % 10 == 0:
            val_loss = validate(model, val_loader, loss_func, args)
            val_losses.append(val_loss)
            if args.save_model:
                if val_loss < best_val_loss:
                    best_model = copy.deepcopy(model)
                    save_model(best_model, model_name, args)
                    save_args(args_name, args)
                    best_val_loss = val_loss


        else:
            #If not the tenth epoch, append the previously calculated loss to the
            #list in order to be able to plot it on the same plot as the training losses
            val_losses.append(val_losses[-1])

        if(epoch%1==0):
            print("train loss", str(round(train_losses[-1], 4)),
                    "val loss", str(round(val_losses[-1], 4)))



    return train_losses, val_losses, best_model


""" def test(loader,device,test_model,
         mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y, is_validation,
          delta_t=0.01, save_model_preds=False, model_type=None):
    raise(NotImplemented)
  
    '''
    Calculates test set losses and validation set errors.
    '''

    loss=0
    velo_rmse = 0
    num_loops=0

    for data in loader:
        data=data.to(device)
        with torch.no_grad():

            #calculate the loss for the model given the test set
            if model_type == 'autoencoder':
                pred = test_model(data, mean_vec_x, std_vec_x)
            else:
                pred = test_model(data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            loss = test_model.loss(pred,data,mean_vec_y,std_vec_y)
            #calculate validation error if asked to
            if (is_validation):

                #Like for the MeshGraphNets model, calculate the mask over which we calculate
                #flow loss and add this calculated RMSE value to our val error
                normal = torch.tensor(0)
                outflow = torch.tensor(5)
                loss_mask = torch.logical_or((torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(0)),
                                             (torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(5)))

                eval_velo = data.x[:, 0:2] + unnormalize( pred[:], mean_vec_y, std_vec_y ) * delta_t
                gs_velo = data.x[:, 0:2] + data.y[:] * delta_t
                
                error = torch.sum((eval_velo - gs_velo) ** 2, axis=1)
                velo_rmse += torch.sqrt(torch.mean(error[loss_mask]))

        num_loops+=1
        # if velocity is evaluated, return velo_rmse as 0
    return loss/num_loops, velo_rmse/num_loops """