#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"
# Some warningsremoval for floating-point round-off errors
prefix="../logs"
export TF_ENABLE_ONEDNN_OPTS=0
python run.py \
    -load_model True \
    -args_file ../logs/args/04-03-24/args_2024_03_04-12.04.json \
    -model_file ../logs/model_chkpoints/04-03-24/model_2024_03_04-12.04/model.pt \
    -make_gif True \
