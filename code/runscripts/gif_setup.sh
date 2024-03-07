#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"
# Some warningsremoval for floating-point round-off errors
prefix="../logs"
day="06-03-24"
date="2024_03_06-17.43_ae_layers-5_latent_dim-128_lr-0.0001_mpl_layers-3_mpl_ratio-0.6"

export TF_ENABLE_ONEDNN_OPTS=0
python run.py \
    -load_model True \
    -args_file ../logs/args/$day/args_$date.json \
    -model_file ../logs/model_chkpoints/$day/model_$date/model.pt \
    -make_gif True \
