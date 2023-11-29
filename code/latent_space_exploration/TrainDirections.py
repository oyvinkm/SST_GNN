import torch
import os
import matplotlib
import sys

from Model.model import make_vae
from deformator import LatentDeformator
from LatentTrainer import Params, Trainer
from loguru import logger
from utils import save_run_params, make_noise

matplotlib.use("Agg")

def save_results_charts(G, deformator, params, out_dir, device):
    deformator.eval()
    G.eval()
    z = make_noise(3, G.dim_z, params.truncation).cuda()
    inspect_all_directions(
        G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(params.shift_scale))),
        zs=z, shifts_r=params.shift_scale, device=device)
    inspect_all_directions(
        G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(3 * params.shift_scale))),
        zs=z, shifts_r=3 * params.shift_scale, device=device)

gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")

torch.set_default_dtype(torch.float32)

logger.debug("Why are we still setting up args this silly way?")
args.deformator = 'proj'
args.directions_count = 100
args.shift_predictor = 'LeNet'
args.out = 'save_location'
args.gen_path = 'gan_log/checkpoint.pt'
args.def_random_init = True
args.gen = 'gan'
args.n_steps = int(3e+3)

save_run_params(args)

G = make_gmsvae(args.model_file)

assert args.directions_count != None

deformator = LatentDeformator(
    shift_dim = args.latent_dim,
    input_dim = args.directions_count,
    out_dim = args.latent_dim,
    type 
)
assert args.shift_predictor != None
if args.shift_predictor == 'ResNet':
    shift_predictor = ResNetShiftPredictor(deformator.input_dim, 1).to(device(args.device))
else:
    shift_predictor = LeNetShiftPredictor(deformator.input_dim, 1).to(device(args.device))

params = Params(**args) # What is this
trainer = Trainer(params, out_dir=args.out, device=args.device)
trainer.train(G, deformator, shift_predictor)

save_results_charts(G, deformator, params, trainer.log_dir, device=args.device)