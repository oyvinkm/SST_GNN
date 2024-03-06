#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"
# Some warningsremoval for floating-point round-off errors
prefix="../logs"
day = "05-03-24"
date="2024_03_05-07.13_ae_layers-3_edge_conv-False_latent_dim-128_mpl_layers-1_num_blocks-3"
export TF_ENABLE_ONEDNN_OPTS=0
python run.py \
    -load_model True \
    -args_file ../logs/args/$day/args_$date.json \
    -model_file ../logs/model_chkpoints/$day/model_$date/model.pt \
    -make_gif True \
