#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"
prefix="../logs"

python DirectionsRun.py \
    -ae_layers 2 \
    -batch_size 16 \
    -edge_conv True \
    -epochs 2000 \
    -hidden_dim 32 \
    -instance_id 1 \
    -latent_space True \
    -loss_step 2 \
    -log_step 10 \
    -lr 1e-4 \
    -latent_dim 128 \
    -logger_lvl DEBUG \
    -loss LMSE \
    -load_model False \
    -mpl_layers 1 \
    -mpl_ratio 0.8 \
    -make_gif False \
    -model_file decoder.pt \
    -num_blocks 1 \
    -normalize False \
    -num_workers 1 \
    -out_feature_dim 11 \
    -opt adam \
    -pool_strat SAG \
    -progress_bar False \
    -random_search False \
    -residual True \
    -save_args_dir $prefix/args/$day \
    -save_gif_dir $prefix/gifs/$day \
    -save_mesh_dir $prefix/meshes/$day \
    -save_model_dir $prefix/model_chkpoints/ \
    -save_plot_dir $prefix/plots/$day \
    -shuffle True \
    -save_plot True \
    -save_model True \
    -save_visual True \
    -save_losses True \
    -save_mesh True \
    -test_ratio 0.1 \
    -transform_p 0.3 \
    -time_stamp $now \
    -transform False \
    -val_ratio 0.1 \
    -weight_decay 0.0005 \