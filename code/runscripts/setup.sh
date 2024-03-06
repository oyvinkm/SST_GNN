#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"
# Some warningsremoval for floating-point round-off errors
prefix="../logs"
export TF_ENABLE_ONEDNN_OPTS=0
python run.py \
    -epochs 40 \
    -random_search False \
    -ae_layers 3 \
    -mpl_ratio .3 \
    -hidden_dim 64 \
    -logger_lvl INFO \
    -loss LMSE \
    -alpha 0.5 \
    -save_plot True \
    -latent_dim 512 \
    -num_blocks 2 \
    -batch_size 2 \
    -mpl_layers 2 \
    -weight_decay 1e-4 \
    -edge_conv True \
    -lr 1e-4 \
    -loss_step 1 \
    -log_step 1 \
