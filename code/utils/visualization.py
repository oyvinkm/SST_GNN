import copy
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap.umap_ as umap
from loguru import logger
from matplotlib import animation
from matplotlib import tri as mtri
from model.decoder import Decoder
from model.utility import LatentVector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.manifold import TSNE
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from utils.helperfuncs import create_folder


def save_plot(args, train_losses, validation_losses):
    loss_name = "loss_" + args.time_stamp
    if not os.path.isdir(args.save_plot_dir):
        os.mkdir(args.save_plot_dir)
    if not os.path.isdir(args.save_loss_over_t_dir):
        os.mkdir(args.save_loss_over_t_dir)
    PATH = os.path.join(args.save_plot_dir, f"{loss_name}.png")
    plot_loss(
        train_loss=train_losses,
        train_label="Training Loss",
        validation_losses=validation_losses,
        val_label="Validation Loss",
        PATH=PATH,
    )


def save_plots(args, losses, test_losses, velo_val_losses):
    """Saves loss plots at args.postprocess_dir"""
    model_name = (
        "model_nl"
        + str(args.num_layers)
        + "_bs"
        + str(args.batch_size)
        + "_hd"
        + str(args.hidden_dim)
        + "_ep"
        + str(args.epochs)
        + "_wd"
        + str(args.weight_decay)
        + "_lr"
        + str(args.lr)
        + "_shuff_"
        + str(args.shuffle)
        + "_tr"
        + str(args.train_size)
        + "_te"
        + str(args.test_size)
    )

    if not os.path.isdir(args.postprocess_dir):
        os.mkdir(args.postprocess_dir)

    PATH = os.path.join(args.postprocess_dir, model_name + ".pdf")

    f = plt.figure()
    plt.title("Losses Plot")
    plt.plot(losses, label="training loss" + " - " + args.model_type)
    plt.plot(test_losses, label="test loss" + " - " + args.model_type)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    f.savefig(PATH, bbox_inches="tight")


def make_animation(
    gs, pred, evl, path, name, skip=1, save_anim=True, plot_variables=False
):
    """
    input gs is a dataloader and each entry contains attributes of many timesteps.

    """
    logger.info("Generating velocity fields...")
    fig, axes = plt.subplots(3, 1, figsize=(20, 16))
    num_steps = len(gs)  # for a single trajectory
    num_frames = num_steps // skip
    logger.info(f"length of trajectory: {num_steps}")

    def animate(num):
        step = (num * skip) % num_steps
        traj = 0

        # gt = next(gs)
        # diff = next(evl)
        # bb_min = gt.x.min()
        # bb_max = gt.x.max()
        # bb_min_evl = diff.x.min()
        # bb_max_evl = diff.x.max()

        bb_min = gs[0].x[:, 0:2].min()  # first two columns are velocity
        bb_max = (
            gs[0].x[:, 0:2].max()
        )  # use max and min velocity of gs dataset at the first step for both
        # gs and prediction plots
        bb_min_evl = evl[0].x[:, 0:2].min()  # first two columns are velocity
        bb_max_evl = (
            evl[0].x[:, 0:2].max()
        )  # use max and min velocity of gs dataset at the first step for both
        # gs and prediction plots
        count = 0

        for ax in axes:
            ax.cla()
            ax.set_aspect("equal")
            ax.set_axis_off()

            pos = gs[step].mesh_pos
            faces = gs[step].cells
            if count == 0:
                # ground truth
                velocity = gs[step].x[:, 0:2]
                title = "Ground truth:"
            elif count == 1:
                # predcition
                velocity = pred[step].x[:, 0:2]
                title = "Reconstruction:"
            else:
                # Reconstruction
                velocity = evl[step].x[:, 0:2]
                title = "Error: (Reconstruction - Ground truth)"

            triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
            if count <= 1:
                # absolute values
                mesh_plot = ax.tripcolor(
                    triang,
                    velocity[:, 0].cpu(),
                    vmin=bb_min,
                    vmax=bb_max,
                    shading="flat",
                )  # x-velocity
                ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
            else:
                # error: (pred - gs)/gs
                mesh_plot = ax.tripcolor(
                    triang,
                    velocity[:, 0].cpu(),
                    vmin=bb_min_evl,
                    vmax=bb_max_evl,
                    shading="flat",
                )  # x-velocity
                ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
                # ax.triplot(triang, lw=0.5, color='0.5')

            ax.set_title(
                "{} Trajectory {} Step {}".format(title, traj, step), fontsize="20"
            )
            # ax.color

            # if (count == 0):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            clb = fig.colorbar(mesh_plot, cax=cax, orientation="vertical")
            clb.ax.tick_params(labelsize=20)

            clb.ax.set_title("x velocity (m/s)", fontdict={"fontsize": 20})
            count += 1
        return (fig,)

    # Save animation for visualization
    if not os.path.exists(path):
        os.makedirs(path)

    if save_anim:
        gs_anim = animation.FuncAnimation(
            fig, animate, frames=num_frames, interval=1000
        )
        writergif = animation.PillowWriter(fps=10)
        anim_path = os.path.join(path, "{}_anim.gif".format(name))
        gs_anim.save(anim_path, writer=writergif)
        plt.show(block=True)
    else:
        pass


def make_gif(model, dataset, args):
    logger.info("Making gif...")
    PRED = copy.deepcopy(dataset)
    GT = copy.deepcopy(dataset)
    DIFF = copy.deepcopy(dataset)
    for pred_data, gt_data, diff_data in zip(PRED, GT, DIFF):
        with torch.no_grad():
            pred, _ = model(Batch.from_data_list([pred_data]).to(args.device))
            pred_data.x = pred.x
            diff_data.x = pred_data.x - gt_data.x.to(args.device)
    logger.info("processing done...")
    gif_name = args.time_stamp
    logger.info(f"saving gif: {gif_name}_anim.gif")

    make_animation(GT, PRED, DIFF, args.save_gif_dir, gif_name, skip=4)
    logger.success("gif complete...")


def make_gif_from_latents(target, shifted, args):
    """makes a gif"""
    logger.info("processing done...")
    folder_path = os.path.join("..", "logs", "direction", "gifs")
    create_folder(folder_path)
    folder_path = os.path.join(folder_path, args.date)
    create_folder(folder_path)
    diff = []
    for pred_data, gt_data in zip(shifted, target):
        res = pred_data.x - gt_data.x
        diff.append(pred_data)
        diff[-1].x = res
    gif_name = args.time_of_the_day
    make_animation(target, shifted, diff, folder_path, gif_name, skip=4)
    logger.success("gif complete")


def draw_graph(g, save=False, args=None):
    """Draws the graph given"""
    G = to_networkx(g, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)
    cent = nx.degree_centrality(G)
    node_size = list(map(lambda x: x * 500, cent.values()))
    cent_array = np.array(list(cent.values()))
    threshold = sorted(cent_array, reverse=True)[10]
    cent_bin = np.where(cent_array >= threshold, 1, 0.1)
    plt.figure(figsize=(12, 12))
    _ = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        cmap=plt.cm.plasma,
        node_color=cent_bin,
        nodelist=list(cent.keys()),
        alpha=cent_bin,
    )
    _ = nx.draw_networkx_edges(G, pos, width=0.25, alpha=0.3)
    if save and args is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        plt.title(f"Graph num nodes: {args.num_nodes}")
        plt.savefig(os.path.join(args.save_dir, f"graph_{args.num_nodes}"))

    plt.show()


def save_mesh(pred, truth, idx, args):
    create_folder(args.save_mesh_dir)
    folder_path = os.path.join(args.save_mesh_dir, args.time_stamp)
    create_folder(folder_path)
    mesh_name = f"mesh_plot_{idx}"
    path = os.path.join(folder_path, mesh_name + ".png")
    # pred.x = pred.x - truth.x
    fig = plot_dual_mesh(pred, truth)
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    logger.success(f"Mesh saved at {path}")


@torch.no_grad()
def plot_mesh(gs, title=None, args=None):
    """plots the graph as a mesh"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    bb_min = gs.x[:, 0:2].min()  # first two columns are velocity
    bb_max = gs.x[
        :, 0:2
    ].max()  # use max and min velocity of gs dataset at the first step for both
    # gs and prediction plots

    ax.cla()
    ax.set_aspect("equal")
    ax.set_axis_off()

    pos = gs.mesh_pos
    faces = gs.cells
    velocity = gs.x[:, 0:2]

    triang = mtri.Triangulation(pos[:, 0].cpu(), pos[:, 1].cpu(), faces.cpu())
    mesh_plot = ax.tripcolor(
        triang, velocity[:, 0].cpu(), vmin=bb_min, vmax=bb_max, shading="flat"
    )  # x-velocity
    ax.triplot(triang, "ko-", ms=0.5, lw=0.3)

    ax.set_title(title, fontsize="20")
    # ax.color

    # if (count == 0):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = fig.colorbar(mesh_plot, cax=cax, orientation="vertical")
    clb.ax.tick_params(labelsize=20)

    clb.ax.set_title("x velocity (m/s)", fontdict={"fontsize": 20})
    return fig


@torch.no_grad()
def plot_dual_mesh(pred_gs, true_gs, title=None, args=None):
    """
    Plots two graphs with each other.
    Can be used to plot the predicted graph and the ground truth
    """
    fig, axes = plt.subplots(2, 1, figsize=(20, 16))
    bb_min = true_gs.x[:, 0:2].min()  # first two columns are velocity
    bb_max = true_gs.x[
        :, 0:2
    ].max()  # use max and min velocity of gs dataset at the first step for both
    # gs and prediction plots

    for idx, ax in enumerate(axes):
        if idx == 0:
            pos = pred_gs.mesh_pos
            faces = pred_gs.cells
            velocity = pred_gs.x[:, 0:2]
            bb_min = pred_gs.x[:, 0:2].min()  # first two columns are velocity
            bb_max = pred_gs.x[
                :, 0:2
            ].max()  # use max and min velocity of gs dataset at the first step for both
            # gs and prediction plots
            title = "Reconstruction"
        elif idx == 1:
            pos = true_gs.mesh_pos
            faces = true_gs.cells
            velocity = true_gs.x[:, 0:2]
            bb_min = true_gs.x[:, 0:2].min()  # first two columns are velocity
            bb_max = true_gs.x[
                :, 0:2
            ].max()  # use max and min velocity of gs dataset at the first step for both
            # gs and prediction plots
            title = "Ground Truth"

        ax.cla()
        ax.set_aspect("equal")
        ax.set_axis_off()

        triang = mtri.Triangulation(pos[:, 0].cpu(), pos[:, 1].cpu(), faces.cpu())
        mesh_plot = ax.tripcolor(
            triang, velocity[:, 0].cpu(), vmin=bb_min, vmax=bb_max, shading="flat"
        )  # x-velocity
        ax.triplot(triang, "ko-", ms=0.5, lw=0.3)

        ax.set_title(title, fontsize="20")
        # ax.color

        # if (count == 0):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = fig.colorbar(mesh_plot, cax=cax, orientation="vertical")
        clb.ax.tick_params(labelsize=20)

        clb.ax.set_title("x velocity (m/s)", fontdict={"fontsize": 20})
    return fig


def plot_loss(
    train_loss=None,
    train_label="Rotate",
    validation_losses=None,
    val_label="One or Two",
    extra_loss=None,
    extra_label="Patches",
    label="Loss",
    title="Loss / Epoch",
    PATH=None,
):
    """
    Takes a list of training and/or validation metrics and plots them
    Returns: plt.figure and ax objects
    """
    if train_loss is None and validation_losses is None:
        raise ValueError(
            "Must specify at least one of 'train_histories' and 'val_histories'"
        )
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    epochs = np.arange(len(train_loss))
    if train_loss is not None:
        ax.plot(
            epochs, train_loss, linewidth=0.8, label=train_label, color="dodgerblue"
        )
    if validation_losses is not None:
        ax.plot(
            epochs, validation_losses, linewidth=0.8, label=val_label, color="darkgreen"
        )
    if extra_loss is not None:
        ax.plot(epochs, extra_loss, linewidth=0.8, label=extra_label, color="darkred")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.legend(loc=0)
    ax.grid(True)
    fig.suptitle(title)
    if PATH is not None:
        plt.savefig(PATH)

    return fig, ax


def plot_test_loss(
    test_loss,
    ts,
    args,
    test_label="validation loss",
    label="Loss",
    title="Loss / T",
    PATH=None,
):
    """
    Takes a list of training and/or validation metrics and plots them
    Returns: plt.figure and ax objects
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    # ax.plot(ts, test_loss, linewidth = .8, label=test_label, color="dodgerblue")
    ax.scatter(ts, test_loss, linewidth=0.8, label=test_label, edgecolors="dodgerblue")
    ax.set_xlabel("t")
    ax.set_ylabel(label)
    ax.legend(loc=0)
    ax.grid(True)
    fig.suptitle(title)
    if PATH is not None:
        create_folder(PATH)
        PATH = os.path.join(PATH, args.time_stamp + ".png")
        plt.savefig(PATH)

    return fig, ax


def visualize_latent_space(latent_time, n_components=2, perplexity=30.0, method="tsne"):
    # Validating input
    # This might change depending on how the data is formatted.
    # latent_time = torch.load('latent_space.pt', map_location='cpu')
    unzipped = [list(t) for t in zip(*latent_time)]
    latent_vectors = np.stack([x.squeeze().detach().numpy() for x in unzipped[0]])
    time_stamps = np.array(unzipped[1])
    assert (
        len(latent_vectors.shape) <= 3 and len(latent_vectors.shape) >= 2
    ), f"Latent vector has dim {len((latent_vectors).shape)}, needs to be on form (no_samples, latent_features, (1))"
    assert (
        len(time_stamps.shape) == 1
    ), f"time_stamps has dim {len(time_stamps.shape)}, need to have shape (no_samples,)"
    if len(latent_vectors.shape) == 3 and latent_vectors.shape[-1] == 1:
        latent_vectors = latent_vectors.squeeze()

    # TSNE settup
    perplexity = min(latent_vectors.shape[0] - 1, perplexity)
    if method == "umap":
        reducer = umap.UMAP()
    else:
        reducer = TSNE(n_components, perplexity=perplexity)
    projection = reducer.fit_transform(latent_vectors.squeeze())
    projection_df = pd.DataFrame(
        {"x": projection[:, 0], "y": projection[:, 1], "label": time_stamps}
    )

    # Plot tsne
    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(1)
    sns.scatterplot(
        x="x", y="y", hue="label", data=projection_df, palette="crest", ax=ax, s=10
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    norm = plt.Normalize(projection_df["label"].min(), projection_df["label"].max())
    sm = plt.cm.ScalarMappable(cmap="crest", norm=norm)
    fig.colorbar(sm, cax=cax, orientation="vertical")

    ax.get_legend().remove()
    ax.set_aspect("equal")
    return fig


@torch.no_grad()
def shift_latents(args, deformator, validation_loader):
    """shifts every latent vector by the given deformator."""
    zs_shifted = []
    target = []
    for i, batch in enumerate(validation_loader):
        z1, z2, z3 = batch[0], batch[1], batch[2]
        deformator.zero_grad()

        # Deformation for 'proj'
        z2_prediction = deformator(z1, z3)
        # scalar_prediction = scaler(z1, z3)

        # z2_prediction = z1 + (direction_prediction * scalar_prediction)
        zs_shifted.extend(list(z2_prediction))
        target.extend(z2)
    return DataLoader(zs_shifted, batch_size=1), DataLoader(target, batch_size=1)


# We want to use dataloaders instead of what we have done.
@torch.no_grad()
def decode_latent_vec(args, decoder, z_shifted_loader):
    """decodes the latent vector given in zs and places them in placeholder
    s.t. they can be shown in a gif"""
    res = []
    for i, z in enumerate(z_shifted_loader):
        z = z.to(args.device)
        z = LatentVector(z, [f"{args.instance_id}"])
        graph = decoder(z)
        res.append(graph.cpu())

    return res


def deformater_visualize(deformator, validation_loader, args):
    """This function decodes a single latent vector and saves it as a graph,
    additionally it makes a gif of what the validation_set would look like
    if it's decoded"""
    PATH = os.path.join(
        args.graph_structure_dir,
        f"{args.instance_id}",
        f"{args.ae_layers}",
    )
    m_ids, m_gs, e_s, m_pos, graph_placeholders = (
        torch.load(os.path.join(PATH, "m_ids.pt")),
        torch.load(os.path.join(PATH, "m_gs.pt")),
        torch.load(os.path.join(PATH, "e_s.pt")),
        torch.load(os.path.join(PATH, "m_pos.pt")),
        torch.load(os.path.join(PATH, "graph_placeholders.pt")),
    )
    decoder = Decoder(args, m_ids, m_gs, e_s, m_pos, graph_placeholders).to(args.device)
    decoder.load_state_dict(torch.load(args.decoder_path))
    if args.decode_single_test:
        # decodes and saves a single graph
        latent_batch = next(iter(validation_loader))[0].to(args.device)
        z = LatentVector(latent_batch, [f"{args.instance_id}"])
        graph_batch = decoder(z)
        graph = Batch.to_data_list(graph_batch)[0]
        save_mesh(graph, graph, "single_clean_graph", args)

    z_shifted_loader, target_loader = shift_latents(args, deformator, validation_loader)

    # length 22
    target_decoded = decode_latent_vec(args, decoder, target_loader)
    # shifted_decoded = decode_latent_vec(args, decoder, z_shifted_loader)

    make_gif_from_latents(target_decoded, target_decoded, args)
