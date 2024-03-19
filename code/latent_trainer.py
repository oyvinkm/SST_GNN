import os

import torch
from loguru import logger
from matplotlib import pyplot as plt
from model.utility import MeanTracker


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


def train(deformator, train_loader, validation_loader, deformator_args):
    deformator.to(deformator_args.device)
    criterion = torch.nn.MSELoss()
    deformator_opt = torch.optim.Adam(deformator.parameters(), lr=1e-4)

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
            z1, z2, z3 = batch[0], batch[1], batch[2]
            deformator.zero_grad()

            # Deformation for 'proj'
            z2_prediction = deformator(z1, z3)

            loss = criterion(z2_prediction, z2)
            loss.backward()

            deformator_opt.step()

            avg_loss.add(loss.item())
            total_loss += loss.item()
        total_loss /= len(train_loader)
        train_losses.append(total_loss)
        val_loss = validate(deformator, validation_loader, epoch)
        val_losses.append(val_loss)
        if epoch % 100 == 0:
            logger.info(f"{epoch=}")
    save_plots(deformator, train_losses, val_losses, deformator_args)


@torch.no_grad()
def validate(deformator, validation_loader, epoch):
    total_loss = 0
    criterion = torch.nn.CosineSimilarity()
    deformator.eval()
    for idx, batch in enumerate(validation_loader):
        z1, z2, z3 = batch[0], batch[1], batch[2]
        deformator.zero_grad()
        z2_prediction = deformator(z1, z3)

        # # Deformation for 'proj'
        # direction_prediction = deformator(z1, z3)
        # scalar_prediction = latent_scaler(z1, z3)

        # z2_prediction = z1 + (direction_prediction * scalar_prediction)

        shift_loss = criterion(z2_prediction, z2)
        # shift_loss = torch.mean(torch.abs(z2_prediction - z2))

        total_loss += shift_loss.item()
    return total_loss / len(validation_loader)


def save_plots(deformator, train_losses, validation_losses, deformator_args):
    """Saves loss plots for deformator at ../logs/direction/plots
    It includes train_losses and validation losses
    """

    PLOTS_PATH = os.path.join("..", "logs", "direction", "plots")
    if not os.path.isdir(PLOTS_PATH):
        os.mkdir(PLOTS_PATH)
    PATH = os.path.join(PLOTS_PATH, f"{deformator_args.time_stamp}.pdf")

    f = plt.figure()
    plt.title("Losses Plot")
    plt.plot(train_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    logger.info(f"Saving loss plot at figure {PATH}")
    plt.legend()
    f.savefig(PATH, bbox_inches="tight")
