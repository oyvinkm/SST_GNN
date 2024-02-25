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
    -mpl_layers 1 \
    -lr 1e-4 \
    -loss_step 1 \
    -log_step 1 \
    -progress_bar False \
    -make_gif False \
    -load_model True \
    #-load_args ../logs/args/22-02-24/args_2024_02_22-09.46.json \
    #-save_model_dir ../logs/model_chkpoints/22-02-24/model_2024_02_22-09.46 \
