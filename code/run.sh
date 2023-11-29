#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"

python random_search.py \
    -ae_layers 3 \
    -batch_size 16 \
    -edge_conv True \
    -epochs 201 \
    -hidden_dim 32 \
    -instance_id 1 \
    -loss_step 50 \
    -log_step 20 \
    -lr 1e-4 \
    -latent_dim 128 \
    -logger_lvl INFO \
    -loss LMSE \
    -load_model False \
    -mpl_layers 1 \
    -mpl_ratio 0.8 \
    -model_file "" \
    -num_blocks 1 \
    -normalize False \
    -num_workers 1 \
    -out_feature_dim 11 \
    -opt adam \
    -pool_strat ASA \
    -progress_bar False \
    -residual True \
    -save_plot_dir plots/$day \
    -save_mesh_dir meshes/$day \
    -save_args_dir args/$day \
    -save_model_dir model_chkpoints/$day \
    -shuffle True \
    -save_plot True \
    -save_model True \
    -save_visual True \
    -save_losses True \
    -save_mesh True \
    -test_ratio 0.1 \
    -transform_p 0.1 \
    -time_stamp $now \
    -transform False \
    -val_ratio 0.1 \
    -weight_decay 0.0005 \
    # -opt_decay_step 30 \
    # -opt_decay_rate 0.1 \
    # -opt_restart 10 \
    # -opt_scheduler step \