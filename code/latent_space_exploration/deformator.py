import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from Models.utils import DeformatorType, torch_expm

class LatentDeformator(nn.Module):
    def __init__(self, shift_dim, input_dim=None, out_dim=None, inner_dim=1024,
                 type=DeformatorType.FC, random_init=False, bias=True):
        super(LatentDeformator, self).__init__()
        self.type = type
        self.shift_dim = shift_dim
        self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
        self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)

        # DEFORMATOR.TYPE == ORTHO AS PROPOSED IN https://arxiv.org/pdf/2207.09740.pdf
        assert self.input_dim == self.out_dim, 'In/out dims must be equal for ortho'
        self.log_mat_half = nn.Parameter((1.0 if random_init else 0.001) * torch.randn(
            [self.input_dim, self.input_dim], device='cuda'), True)
        
    
    def forward(self, input):
        input = input.view([-1, self.input_dim])
        mat = torch_expm((self.log_mat_half - self.log_mat_half.transpose(0, 1)).unsqueeze(0))
        out = F.linear(input, mat)
        flat_shift_dim = np.product(self.shift_dim)
        if out.shape[1] < flat_shift_dim:
            padding = torch.zeros([out.shape[0], flat_shift_dim - out.shape[1]], device=out.device)
            out = torch.cat([out, padding], dim=1)
        elif out.shape[1] > flat_shift_dim:
            out = out[:, :flat_shift_dim]

        # handle spatial shifts
        try:
            out = out.view([-1] + self.shift_dim)
        except Exception:
            pass

        return out.reshape(-1,self.out_dim,1,1)

def normal_projection_stat(x):
    x = x.view([x.shape[0], -1])
    direction = torch.randn(x.shape[1], requires_grad=False, device=x.device)
    direction = direction / torch.norm(direction)
    projection = torch.matmul(x, direction)

    std, mean = torch.std_mean(projection)
    return std, mean