#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"
# Some warningsremoval for floating-point round-off errors
prefix="../logs"
export TF_ENABLE_ONEDNN_OPTS=0
python run.py \
    -epochs 30 \
    -random_search False \
    -ae_layers 5 \
    -hidden_dim 32 \
    -latent_dim 256 \
    -logger_lvl INFO \
    -loss LMSE \
    -lr 1e-3 \
    -alpha 0.5 \
    -save_plot True \
    -pool_strat TopK \
    -num_blocks 2 \
    -batch_size 2 \
    -weight_decay 1e-6 \
    -edge_conv True \
    -loss_step 2 \
    -log_step 2 \
