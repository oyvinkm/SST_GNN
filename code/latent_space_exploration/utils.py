from enum import Enum
import numpy as np
import os
import json
import types
from functools import wraps
import io
from PIL import Image
import matplotlib.pyplot as plt
import sys

import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torchvision.transforms import Resize


class ShiftDistribution(Enum):
    """used by another script"""
    NORMAL = 0,
    UNIFORM = 1,

class DeformatorType(Enum):
    """used by another script"""
    ORTHO = 4 #probably the only deformatortype we need.


SHIFT_DISTRIDUTION_DICT = {
    'normal': ShiftDistribution.NORMAL,
    'uniform': ShiftDistribution.UNIFORM,
    None: None
}

def torch_expm(A):
    """used by another script"""
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1, 2), keepdim=True))

    # Scaling step
    maxnorm = torch.tensor([5.371920351148152], dtype=A.dtype, device=A.device)
    zero = torch.tensor([0.0], dtype=A.dtype, device=A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    A_scaled = A / 2.0 ** n_squarings
    n_squarings = n_squarings.flatten().type(torch.int64)

    # Pade 13 approximation
    U, V = torch_pade13(A_scaled)
    P = U + V
    Q = -U + V
    R, _ = torch.solve(P, Q)

    # Unsquaring step
    res = [R]
    for i in range(int(n_squarings.max())):
        res.append(res[-1].matmul(res[-1]))
    R = torch.stack(res)
    expmA = R[n_squarings, torch.arange(n_A)]
    return expmA[0]

def make_noise(batch, dim, truncation=None):
    """used by another script"""
    if isinstance(dim, int):
        dim = [dim]
    if truncation is None or truncation == 1.0:
        return torch.randn([batch] + dim+[1,1])
    else:
        return torch.from_numpy(truncated_noise([batch] + dim, truncation)).to(torch.float)

class MeanTracker(object):
    """used by another script"""
    def __init__(self, name):
        self.values = []
        self.name = name

    def add(self, val):
        self.values.append(float(val))

    def mean(self):
        return np.mean(self.values)

    def flush(self):
        mean = self.mean()
        self.values = []
        return self.name, mean


def save_run_params(args):
    """used by another script"""
    os.makedirs(args['out'], exist_ok=True)
    with open(os.path.join(args['out'], 'args.json'), 'w') as args_file:
        json.dump(args, args_file)

@torch.no_grad()
def make_interpolation_chart(G, deformator=None, z=None,
                             shifts_r=10.0, shifts_count=5,
                             dims=None, dims_count=10, texts=None, device='cpu', **kwargs):
    with_deformation = deformator is not None
    if with_deformation:
        deformator_is_training = deformator.training
        deformator.eval()
    z = z if z is not None else make_noise(1, G.dim_z).to(device)

    if with_deformation:
        original_img = G(z).cpu()
    else:
        original_img = G(z).cpu()
    imgs = []
    if dims is None:
        dims = range(dims_count)
    for i in dims:
        imgs.append(interpolate(G, z, shifts_r, shifts_count, i, deformator, device=device))

    rows_count = len(imgs) + 1
    fig, axs = plt.subplots(rows_count, **kwargs)

    axs[0].axis('off')
    axs[0].imshow(to_image(original_img, True))

    if texts is None:
        texts = dims
    for ax, shifts_imgs, text in zip(axs[1:], imgs, texts):
        ax.axis('off')
        plt.subplots_adjust(left=0.5)
        ax.imshow(to_image(make_grid(shifts_imgs, nrow=(2 * shifts_count + 1), padding=1), True))
        ax.text(-20, 21, str(text), fontsize=10)

    if deformator is not None and deformator_is_training:
        deformator.train()

    return fig

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


