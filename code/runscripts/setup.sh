#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"
# Some warningsremoval for floating-point round-off errors
prefix="../logs"
export TF_ENABLE_ONEDNN_OPTS=0
python run.py \
    -epochs 1 \
    -ae_layers 2 \
    -hidden_dim 64 \
    -batch_size 8 \
    -mpl_layers 2 \
    -loss_step 1 \
    -log_step 5 \
    -make_gif False \
    -load_model False \
