"""
    Training/testing file with funcions train(), test(), validate()
"""
import copy
import os

import numpy as np
import torch
import enlighten
from loguru import logger
from torch.nn import MSELoss
from tqdm import trange
from mask import AttributeMask
from torch_geometric import transforms as T
from matplotlib import pyplot as plt

from utils.visualization import plot_dual_mesh


def save_model(best_model, args):
    if not os.path.isdir(args.save_model_dir):
        os.mkdir(args.save_model_dir)
    model_name = "model_" + args.time_stamp
    path = os.path.join(args.save_model_dir, model_name + ".pt")
    torch.save(best_model.state_dict(), path)


def save_args(args):
    if not os.path.isdir(args.save_args_dir):
        os.mkdir(args.save_args_dir)
    args_name = "args_" + args.time_stamp
    path = os.path.join(args.save_args_dir, args_name + ".txt")
    with open(path, "w") as f:
        for key, value in args.__dict__.items():
            f.write("%s: %s\n" % (key, value))


def save_mesh(pred, truth, idx, args):
    if not os.path.isdir(args.save_mesh_dir):
        os.mkdir(args.save_mesh_dir)
    mesh_name = f"mesh_plot_{idx}_{args.time_stamp}"
    path = os.path.join(args.save_mesh_dir, mesh_name + ".png")
    fig = plot_dual_mesh(pred, truth)
    fig.savefig(path, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def validate(model, val_loader, loss_func, epoch, args):
    """
    Performs a validation run on our current model with the validationset
    saved in the val_loader.
    """
    total_loss = 0
    model.eval()
    for idx, batch in enumerate(val_loader):
        # data = transform(batch).to(args.device)
        # Note that normalization must be done before it's called. The unnormalized
        # data needs to be preserved in order to correctly calculate the loss
        batch = batch.to(args.device)
        b_data = batch.clone()
        b_data = transform_batch(batch, args)
        pred, _ = model(b_data)
        loss = loss_func(pred.x, batch.x)
        total_loss += loss.item()
        if idx == 0 and args.save_mesh:
            save_mesh(pred, batch, epoch, args)

    total_loss /= idx
    return total_loss


@torch.no_grad()
def test(model, test_loader, loss_func, args):
    """
    Performs a test run on our final model with the test
    saved in the test_loader.
    """
    if loss_func is None:
        loss_func = MSELoss()
    total_loss = 0
    model.eval()
    for idx, batch in enumerate(test_loader):
        # data = transform(batch).to(args.device)
        # Note that normalization must be done before it's called. The unnormalized
        # data needs to be preserved in order to correctly calculate the loss
        batch = batch.to(args.device)
        b_data = transform_batch(batch, args)
        pred, _ = model(b_data)
        if idx == 0 and args.save_mesh:
            save_mesh(pred, batch, 1, args)
        loss = loss_func(pred.x, batch.x)
        total_loss += loss.item()

    total_loss /= idx
    return total_loss

def transform_batch(b_data, args):
    if args.transform == 'Attribute':
        transforms = T.Compose([AttributeMask(args.transform_p)])
        trsfmd = transforms(b_data.clone())
        return trsfmd
    else:
        return b_data.clone()

def train(model, train_loader, val_loader, optimizer, args):
    """
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    """
    model = model.to(args.device)

    # TODO: Save args in a file with date and time

    # train
    # NOTE: Might make dependent on args which loss function
    loss_func = MSELoss()
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_model = None

    manager = enlighten.get_manager()
    if args.progress_bar:
        epochs = manager.counter(total=args.epochs, desc="Epochs", unit="Epochs", color="red")
    for epoch in range(args.epochs):
        if args.progress_bar:
            epochs.update()
            batch_counter = manager.counter(total=len(train_loader), desc="Batches", unit="Batches", color="blue", leave=False, position=True)
        total_loss = 0
        model.train()
        for idx, batch in enumerate(train_loader):
            if args.progress_bar:
                batch_counter.update()
            # data = transform(batch).to(args.device)
            # Note that normalization must be done before it's called. The unnormalized
            # data needs to be preserved in order to correctly calculate the loss
            batch = batch.to(args.device)
            logger.debug(f'batch : {batch}')
            logger.debug(f'Data before transformation, true mean : {np.sum(batch.x.detach().cpu().numpy())}, transformed mean : {np.sum(batch.x.detach().cpu().numpy())} ')
            b_data = transform_batch(batch, args)
            logger.debug(f'Data transformed, true mean : {np.mean(batch.x.detach().cpu().numpy())}, transformed mean : {np.mean(b_data.x.detach().cpu().numpy())} ')
            optimizer.zero_grad()  # zero gradients each time
            pred, _ = model(b_data)
            # NOTE: Does the loss have to be a function in the model?
            loss = loss_func(pred.x, batch.x)
            loss.backward()  # backpropagate loss
            optimizer.step()
            total_loss += loss.item()
        if args.progress_bar:
            batch_counter.close()
        logger.debug(f'Len loader: {len(train_loader)}')
        logger.debug(f"Index : {idx}")
        total_loss /= len(train_loader)
        train_losses.append(total_loss)

        # Every tenth epoch, calculate acceleration test loss and velocity validation loss
        if epoch % 2 == 0:
            val_loss = validate(model, val_loader, loss_func, epoch, args)
            val_losses.append(val_loss)
            if args.save_model:
                if val_loss < best_val_loss:
                    best_model = copy.deepcopy(model)
                    save_model(best_model, args)
                    save_args(args)
                    best_val_loss = val_loss

        else:
            # If not the tenth epoch, append the previously calculated loss to the
            # list in order to be able to plot it on the same plot as the training losses
            val_losses.append(val_losses[-1])

        if epoch % 1 == 0:
            logger.info(f"Loss Epoch_{epoch}:\n\
                        Training Loss : {round(train_losses[-1], 4)}\n\
                        Validation Loss : {round(val_losses[-1], 4)}")
    if args.progress_bar:
        epoch.stop()
    return train_losses, val_losses, best_model
