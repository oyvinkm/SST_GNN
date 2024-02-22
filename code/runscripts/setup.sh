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
    -latent_dim 128 \
    -batch_size 4 \
    -mpl_layers 2 \
    -lr 1e-4 \
    -loss_step 1 \
    -log_step 1 \
    -make_gif False \
    -load_model False \
