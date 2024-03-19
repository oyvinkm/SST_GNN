import torch
import torch.nn as nn
from loguru import logger  # noqa: F401


class LatentDeformator(nn.Module):
    """Deforms the input in latent space, it might be applying the shift.
    In addition it explores the amount of directions given in direction_args['directions_count']
    """

    def __init__(
        self,
        input_dim=None,
        out_dim=None,
    ):
        super(LatentDeformator, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.deform1 = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 2, bias=True),
            nn.SELU(),
            nn.Linear(self.input_dim * 2, self.input_dim, bias=True),
            nn.SELU(),
            nn.Linear(self.input_dim, self.out_dim, bias=True),
        )

        self.deform2 = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 2, bias=True),
            nn.SELU(),
            nn.Linear(self.input_dim * 2, self.input_dim, bias=True),
            nn.SELU(),
            nn.Linear(self.input_dim, self.out_dim, bias=True),
        )

        self.collect = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim * 4, bias=True),
            nn.BatchNorm1d(self.input_dim * 4),
            nn.SELU(),
            nn.Linear(self.input_dim * 4, self.input_dim * 2, bias=True),
            nn.SELU(),
            nn.Linear(self.input_dim * 2, self.out_dim, bias=True),
        )

        self.simple = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim, bias=True),
            # nn.BatchNorm1d(self.input_dim),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(self.input_dim, self.out_dim, bias=True),
        )

        self.simple1 = nn.Linear(self.input_dim * 2, self.input_dim, bias=True)
        # self.simple2 =nn.BatchNorm1d(self.input_dim)
        self.simple2 = nn.Dropout(p=0.9)
        self.simple3 = nn.SELU()
        self.simple4 = nn.Linear(self.input_dim, self.out_dim, bias=True)
        # As used in Latent-Space-Exploration-CT
        # for layer in [self.lin1, self.lin2]:
        #     layer.weight.data = torch.zeros_like(layer.weight.data)
        #     layer.weight.data[:shift_dim, :shift_dim] = torch.eye(shift_dim)
        #     layer.weight.data = 0.1 * torch.randn_like(layer.weight.data)

    def forward(self, z1, z3):
        in1 = z1.clone()
        in2 = z3.clone()
        # z1 = self.deform1(in1)
        # z3 = self.deform2(in2)
        z = torch.cat([in1, in2], dim=2)
        # out = self.collect(z)
        logger.debug(f"{z.shape}")
        out = self.simple1(z)
        logger.debug(f"{out.shape}")
        out = self.simple2(out)
        out = self.simple3(out)
        out = self.simple4(out)
        out = out.reshape(-1, 1, self.out_dim)
        return out


class LatentScaler(nn.Module):
    def __init__(self, input_dim):
        super(LatentScaler, self).__init__()
        self.input_dim = input_dim

        self.shift = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2, bias=True),
            nn.BatchNorm1d(self.input_dim // 2),
            nn.SELU(),
            nn.Linear(self.input_dim // 2, 1, bias=True),
        )

    def forward(self, input):
        out = self.shift(input.squeeze(dim=1))
        return out.reshape(out.shape[0], 1, 1)
