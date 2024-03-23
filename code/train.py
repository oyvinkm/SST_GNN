"""
    Training/testing file with funcions train(), test(), validate()
"""

import copy
import json
import os
from random import randint

import enlighten
import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.nn import MSELoss
from torch_geometric import transforms as T
from utils.opt import build_optimizer
from utils.transforms import AttributeMask, FlipGraph
from utils.visualization import save_mesh


def train(model, train_loader, val_loader, args):
    """
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    """
    model = model.to(args.device)
    optimizer = build_optimizer(args, model.parameters())
    if args.loss == "LMSE":
        criterion = LMSELoss()
        logger.info("Loss is LMSE")
    else:
        criterion = MSELoss()
        logger.info("Loss is MSE")

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_model = None
    beta = 1e-3
    alpha = args.alpha
    if args.progress_bar:
        manager = enlighten.get_manager()
        epochs = manager.counter(
            total=args.epochs, desc="Epochs", unit="Epochs", color="red"
        )
    logger.success(f"Beginning training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if args.progress_bar:
            epochs.update()
            batch_counter = manager.counter(
                total=len(train_loader),
                desc="Batches",
                unit="Batches",
                color="blue",
                leave=False,
                position=True,
            )
        total_loss = 0
        model.train()
        for idx, batch in enumerate(train_loader):
            if args.progress_bar:
                batch_counter.update()
            # data = transform(batch).to(args.device)
            # Note that normalization must be done before it's called. The unnormalized
            # data needs to be preserved in order to correctly calculate the loss
            optimizer.zero_grad()  # zero gradients each time
            batch = batch.to(args.device)
            b_data = batch.clone()
            pred, kl = model(b_data)
            mask = batch.x[:, 0] < 0.0
            rec_loss_node = criterion(pred.x, batch.x)
            mask_loss_node = criterion(pred.x[mask], batch.x[mask])
            loss = beta * kl + rec_loss_node + alpha * mask_loss_node
            if loss.isnan():
                logger.debug(f"Loss has become NaN after {epoch} epochs")
                torch.save(pred.to("cpu"), "NaN_tensor.pt")
            try:
                loss.backward()  # backpropagate loss
            except RuntimeError as e:
                logger.error(f"{e}")
                return train_losses, val_losses, best_model

            optimizer.step()
            total_loss += loss.item()
        if args.progress_bar:
            batch_counter.close()
        total_loss /= len(train_loader)
        train_losses.append(total_loss)
        # Every tenth epoch, calculate acceleration test loss and velocity validation loss
        if epoch % args.log_step == 0:
            val_loss = validate(model, val_loader, criterion, epoch, args)
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

        if epoch % args.loss_step == 0:
            logger.success(
                f"Loss Epoch_{epoch}:\n\
                        Training Loss : {round(train_losses[-1], 4)}\n\
                        Validation Loss : {round(val_losses[-1], 4)}"
            )
    if args.progress_bar:
        epochs.close()
        manager.stop()

    return train_losses, val_losses, best_model


@torch.no_grad()
def validate(model, val_loader, criterion, epoch, args):
    """
    Performs a validation run on our current model with the validationset
    saved in the val_loader.
    """
    total_loss = 0
    model.eval()

    early = False
    late = False
    for idx, batch in enumerate(val_loader):
        batch = batch.to(args.device)
        # batch.x = F.normalize(batch.x)
        b_data = batch.clone()
        pred, _ = model(b_data, Train=False)
        rec_loss_node = criterion(pred.x, batch.x)
        # rec_loss_edge = criterion(pred.edge_attr, batch.edge_attr)
        loss = rec_loss_node
        total_loss += loss.item()
        if batch.t < 10 and not early:
            save_mesh(pred, batch, f"{epoch}_{batch.t.item()}", args)
            early = True
        if batch.t < 100 and not late:
            save_mesh(pred, batch, f"{epoch}_{batch.t.item()}", args)
            late = True
        if idx == len(val_loader) - 1 and not (early or late):
            save_mesh(pred, batch, f"{epoch}_{batch.t.item()}", args)
    total_loss /= idx
    return total_loss


@torch.no_grad()
def loss_over_t(model, val_loader, args):
    """
    Performs a test run on our final model with the test
    saved in the val_loader.
    """
    logger.debug("======= TESTING =======")

    loss_over_t = []
    ts = []
    criterion = MSELoss()
    model.eval()
    for idx, batch in enumerate(val_loader):
        batch = batch.to(args.device)
        b_data = batch.clone()
        pred, _ = model(b_data, Train=False)
        loss = criterion(pred.x[:, :2], batch.x[:, :2])
        loss_over_t.append(loss.item())
        ts.append(int(batch.t.cpu()[0]))
    return ts, loss_over_t


def augment_batch(b_data):
    augment = randint(0, 1)
    if augment:
        augments = T.Compose([FlipGraph()])
        b_data = augments(b_data.clone())
    return b_data


def transform_batch(b_data, args):
    if args.transform:
        transforms = T.Compose([AttributeMask(args.transform_p, args.device)])
        trsfmd = transforms(b_data.clone())
        return trsfmd
    else:
        return b_data.clone()


def save_model(best_model, args):
    """Saves the model and the decoder in a folder"""
    if not os.path.isdir(args.save_model_dir):
        os.mkdir(args.save_model_dir)
    model_name = "model_" + args.time_stamp
    final_place = os.path.join(args.save_model_dir, model_name)
    if not os.path.isdir(final_place):
        os.mkdir(final_place)

    decoder_path = os.path.join(final_place, "decoder.pt")
    encoder_path = os.path.join(final_place, "encoder.pt")
    model_path = os.path.join(final_place, "model.pt")
    torch.save(best_model.decoder.state_dict(), decoder_path)
    torch.save(best_model.encoder.state_dict(), encoder_path)
    torch.save(best_model.state_dict(), model_path)


def save_args(args):
    # logger.debug(f'{args.__dict__=}')
    if not os.path.isdir(args.save_args_dir):
        os.mkdir(args.save_args_dir)
    args_name = "args_" + args.time_stamp
    path = os.path.join(args.save_args_dir, args_name + ".json")
    with open(path, "w") as f:
        json.dump(args.__dict__, f)


# NOTE: MIGHT NOT BE A GOOD IDEA TO REMOVE +1 FROM LOSS
class LMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        loss_mask = torch.where(torch.argmax(actual[:, 2:7], dim=1) != 3)
        return torch.log(
            self.mse(pred[loss_mask][:, :2], actual[loss_mask][:, :2])
        )  # +1 to keep the loss from under 0
