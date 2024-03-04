#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"
# Some warningsremoval for floating-point round-off errors
prefix="../logs"
export TF_ENABLE_ONEDNN_OPTS=0
python run.py \
    -epochs 50 \
    -ae_layers 3 \
    -hidden_dim 64 \
    -logger_lvl INFO \
    -progress_bar True \
    -loss LMSE \
    -alpha 1. \
    -save_plot True \
    -latent_dim 128 \
    -num_blocks 2 \
    -batch_size 8 \
    -mpl_layers 2 \
    -lr 1e-4 \
    -loss_step 1 \
    -log_step 1 \
