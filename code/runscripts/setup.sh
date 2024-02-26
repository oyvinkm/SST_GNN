#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"
# Some warningsremoval for floating-point round-off errors
prefix="../logs"
export TF_ENABLE_ONEDNN_OPTS=0
python run.py \
    -epochs 10 \
    -ae_layers 3 \
    -hidden_dim 64 \
    -loss MSE \
    -alpha 1. \
    -save_plot True \
    -latent_dim 256 \
    -batch_size 4 \
    -mpl_layers 2 \
    -lr 1e-5 \
    -loss_step 1 \
    -log_step 1 \
    #-load_model True \
    #-model_file ../logs/model_chkpoints/25-02-24/model_2024_02_25-16.25/model.pt \
    #-make_gif True \
