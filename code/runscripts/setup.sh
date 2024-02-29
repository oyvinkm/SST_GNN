#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"
# Some warningsremoval for floating-point round-off errors
prefix="../logs"
export TF_ENABLE_ONEDNN_OPTS=0
python run.py \
    -epochs 50 \
    -ae_layers 4 \
    -hidden_dim 128 \
    -logger_lvl INFO \
    -progress_bar True \
    -loss LMSE \
    -alpha 1. \
    -save_plot True \
    -latent_dim 256 \
    -batch_size 4 \
    -mpl_layers 2 \
    -lr 1e-4 \
    -loss_step 1 \
    -log_step 1 \
    #-load_model True \
    #-args_file ../logs/args/28-02-24/args_2024_02_28-19.59.json \
    #-model_file ../logs/model_chkpoints/28-02-24/model_2024_02_28-19.59/model.pt \
    #-make_gif True \
