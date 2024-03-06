import os

import torch
import torch.nn as nn
from loguru import logger
from matplotlib import pyplot as plt

from model.utility import DeformatorType, MeanTracker, ShiftDistribution


def save_difference_norms(train_set):
    norms = []
    for z1, z2 in train_set:
        res = torch.norm(z2 - z1)
        norms.append(res.item())
    PLOTS_PATH = os.path.join("..", "logs", "direction", "norms")
    if not os.path.isdir(PLOTS_PATH):
        os.mkdir(PLOTS_PATH)
    PATH = os.path.join(PLOTS_PATH, "difference_norms.pdf")

    f = plt.figure()
    plt.title("Difference Norms Plot")
    plt.plot(norms)  # , label="norm of difference between z1 and z2")
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("norm of vector from z1 to z2")

    plt.legend()
    f.savefig(PATH, bbox_inches="tight")


def train(deformator, latent_scaler, train_loader, validation_loader, deformator_args):
    deformator.to(self.device).train()

    logger.debug(f"look two lines below {deformator.type=}")

    deformator_opt = torch.optim.Adam(deformator.parameters(), lr=1-e4)

    avgs = (
        MeanTracker("percent"),
        MeanTracker("loss"),
        MeanTracker("direction_loss"),
        MeanTracker("shift_loss"),
    )
    avg_correct_percent, avg_loss, avg_label_loss, avg_shift_loss = avgs

    train_losses = []
    val_losses = []
    for epoch in range(deformator_args.epochs):
        # It's approximately 2 seconds for 100 epochs
        total_loss = 0
        for idx, batch in enumerate(train_loader):
            z1, z2 = batch

            deformator.zero_grad()

            # Deformation for 'proj'
            direction_prediction = deformator(z1).squeeze(dim = 3)
            scalar_prediction = latent_scaler(z1)

            
            z2_prediction = z1 + (scalar_prediction * shift_prediction).squeeze(dim=2)

            loss = torch.mean(torch.abs(z2_prediction - z2))
            loss.backward()

            deformator_opt.step()

            avg_loss.add(loss.item())
            total_loss += loss.item()
        total_loss /= len(train_loader)
        train_losses.append(total_loss)
        val_loss = self.validate(deformator, validation_loader, epoch)
        val_losses.append(val_loss)
    save_plots(deformator, train_losses, val_losses, deformator_args)

@torch.no_grad()
def validate(deformator, latent_scaler, validation_loader, epoch):
    total_loss = 0
    deformator.eval()
    for idx, batch in enumerate(validation_loader):
        z1, z2 = batch

        # Deformation for 'proj'
        # shift_prediction = deformator(z1).squeeze(dim = 3)

        # Deformation for 'id'

        direction_prediction = deformator(z1).squeeze(dim = 3)
        scalar_prediction = latent_scaler(z1)

        z2_prediction = z1 + (scalar_prediction * shift_prediction).squeeze(dim=2)

        shift_loss = torch.mean(torch.abs(z2_prediction - z2))

        total_loss += shift_loss.item()
    return total_loss / len(validation_loader)


def save_plots(deformator, train_losses, validation_losses, deformator_args):
    """Saves loss plots for deformator at ../logs/direction/plots
    It includes train_losses and validation losses
    """

    PLOTS_PATH = os.path.join("..", "logs", "direction", "plots")
    if not os.path.isdir(PLOTS_PATH):
        os.mkdir(PLOTS_PATH)
    PATH = os.path.join(
        PLOTS_PATH, f"{deformator.type=}_{deformator_args.time_stamp}.pdf"
    )

    f = plt.figure()
    plt.title("Losses Plot")
    plt.plot(train_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    f.savefig(PATH, bbox_inches="tight")
