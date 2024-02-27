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
from torch.nn import functional as F
from torch_geometric import transforms as T
from utils.opt import build_optimizer
from utils.transforms import AttributeMask, FlipGraph
from utils.visualization import save_mesh


def train(model, train_loader, validation_loader, args):
    """
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    """
    logger.success("Started training...")
    model = model.to(args.device)
    optimizer = build_optimizer(args, model.parameters())
    criterion = LMSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_model = None
    beta = 0.01
    if args.progress_bar:
        manager = enlighten.get_manager()
        epochs = manager.counter(
            total=args.epochs, desc="Epochs", unit="Epochs", color="red"
        )
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
            batch = batch.to(args.device)
            batch.x = F.normalize(batch.x)
            batch.edge_attr = F.normalize(batch.edge_attr)
            b_data = transform_batch(batch, args)
            # b_data = augment_batch(b_data)
            optimizer.zero_grad()  # zero gradients each time
            pred, kl = model(b_data)
            rec_loss = criterion(pred.x[:, :2], batch.x[:, :2])
            loss = beta * kl + rec_loss
            loss.backward()  # backpropagate loss
            optimizer.step()
            total_loss += loss.item()
        if args.progress_bar:
            batch_counter.close()
        total_loss /= len(train_loader)
        train_losses.append(total_loss)
        # Every tenth epoch, calculate acceleration test loss and velocity validation loss
        if epoch % args.log_step == 0:
            val_loss = validate(model, validation_loader, criterion, epoch, args)
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
    logger.success("Training and validation complete...")
    return train_losses, val_losses, best_model


@torch.no_grad()
def validate(model, validation_loader, criterion, epoch, args):
    """
    Performs a validation run on our current model with the validationset
    saved in the validation_loader.
    """
    random_idx = randint(
        0, len(100)
    )  # doing the first 50 as these varies a lot from the mean graph
    total_loss = 0
    model.eval()
    for idx, batch in enumerate(validation_loader):
        logger.debug(f"{idx=}")
        # data = transform(batch).to(args.device)
        # Note that normalization must be done before it's called. The unnormalized
        # data needs to be preserved in order to correctly calculate the loss
        batch = batch.to(args.device)
        batch.x = F.normalize(batch.x)
        batch.edge_attr = F.normalize(batch.edge_attr)
        b_data = transform_batch(batch, args)
        b_data = batch.clone()
        pred, _ = model(b_data, Train=False)
        loss = criterion(pred.x[:, :2], b_data.x[:, :2])
        total_loss += loss.item()
        if idx == random_idx and args.save_mesh:
            save_mesh(pred, batch, epoch, args)
    total_loss /= idx
    return total_loss


@torch.no_grad()
def test(model, test_loader, args):
    """
    Performs a test run on our final model with the test
    saved in the test_loader.
    """
    logger.debug("======= TESTING =======")
    kld = nn.KLDivLoss(reduction="batchmean")

    loss_over_t = []
    ts = []
    criterion = MSELoss()
    total_loss = 0
    total_accuracy = 0
    model.eval()
    for idx, batch in enumerate(test_loader):
        # data = transform(batch).to(args.device)
        # Note that normalization must be done before it's called. The unnormalized
        # data needs to be preserved in order to correctly calculate the loss
        batch = batch.to(args.device)
        batch.x = F.normalize(batch.x)
        batch.edge_attr = F.normalize(batch.edge_attr)
        # Needs to clone as operations happens in place
        b_data = batch.clone()
        logger.error(f"{b_data=}")
        pred, _ = model(b_data, Train=False)
        if idx == 0 and args.save_mesh:
            save_mesh(pred, batch, "test", args)
        loss = criterion(pred.x[:, :2], batch.x[:, :2])
        total_loss += loss.item()
        total_accuracy += kld(input=torch.log(pred.x), target=batch.x).item()
        loss_over_t.append(loss.item())
        ts.append(batch.t.cpu())

    total_loss /= idx
    total_accuracy /= idx
    save_accuracy(total_accuracy, args)
    return total_loss, loss_over_t, ts


def save_accuracy(accuracy, args):
    if not os.path.isdir(args.save_accuracy_dir):
        os.mkdir(args.save_accuracy_dir)
    filename = "accuracy_" + args.time_stamp
    path = os.path.join(args.save_accuracy_dir, filename + ".txt")
    with open(path, "w") as f:
        f.write("accuracy: %s" % (accuracy))
    f.close()


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


class LMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.log(self.mse(pred, actual) + 1)  # +1 to keep the loss from under 0
