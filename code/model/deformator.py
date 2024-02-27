import torch
import torch.nn as nn
from loguru import logger


class LatentDeformator(nn.Module):
    """Deforms the input in latent space, it might be applying the shift.
    In addition it explores the amount of directions given in direction_args['directions_count']
    """

    def __init__(
        self,
        shift_dim,
        input_dim=None,
        out_dim=None,
    ):
        super(LatentDeformator, self).__init__()
        self.shift_dim = shift_dim
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.lin1 = nn.Linear(self.input_dim, self.input_dim * 2, bias=True)
        self.lin2 = nn.Linear(self.input_dim * 2, self.out_dim, bias=True)
        self.lrel1 = nn.LeakyReLU()

        for layer in [self.lin1, self.lin2]:
            self.layer.weight.data = torch.zeros_like(self.layer.weight.data)
            self.layer.weight.data[:shift_dim, :shift_dim] = torch.eye(shift_dim)
            self.layer.weight.data = 0.1 * torch.randn_like(self.layer.weight.data)

    def forward(self, input):
        input = input.view([-1, self.input_dim])

        input_norm = torch.linalg.vector_norm(input, dim=1, keepdim=True)
        out = self.lin1(input)
        out = self.lrel1(out)
        out = self.lin2(out)
        out = (input_norm / torch.norm(out, dim=1, keepdim=True)) * out

        logger.debug(f"{torch.norm(out)=}")
        assert (
            torch.linalg.vector_norm(out) == 1.0
        ), "The output of the deformator is not a directions vector"

        # try:
        #     out = out.view([-1] + self.shift_dim)
        # except Exception:
        #     pass

        return out.reshape(-1, self.out_dim, 1, 1)


class LatentScaler(nn.Module):
    def __init__(self, input_dim):
        self.input_dim = input_dim

        self.lin1 = nn.Linear(self.input_dim, self.input_dim / 2, bias=True)
        self.relu1 = nn.relu()
        self.lin2 = nn.Linear(self.input_dim / 2, 1, bias=True)
        self.relu2 = nn.relu()

    def forward(self, input):
        out = self.lin1(input)
        out = self.relu1(out)
        out = self.lin2(out)
        out = self.relu2(out)
        return out
