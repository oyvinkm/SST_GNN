#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"

python random_search.py \
    -loss_step 50 \
    -log_step 20 \
    -instance_id 1 \
    -batch_size 16 \
    -out_feature_dim 11 \
    -epochs 101 \
    -ae_layers 4 \
    -hidden_dim 64 \
    -mpl_layers 1 \
    -num_blocks 1 \
    -latent_dim 128 \
    -opt_decay_step 30 \
    -opt_restart 10 \
    -num_workers 1 \
    -test_ratio 0.1 \
    -val_ratio 0.1 \
    -mpl_ratio 0.8 \
    -opt_decay_rate 0.1 \
    -transform_p 0.1 \
    -weight_decay 0.0005 \
    -lr 1e-4 \
    -logger_lvl INFO \
    -ae_pool_strat SAG \
    -pool_strat ASA \
    -opt adam \
    -loss LMSE \
    -opt_scheduler step \
    -model_file model_2023_11_16-17.16_ae_layers-2_ae_ratio-0.1_hidden_dim-64_latent_dim-128_loss-LMSE_lr-0.0001_mpl_layers-2_mpl_ratio-0.8_num_blocks-1_pool_strat-ASA.pt \
    -save_plot_dir plots/$day \
    -save_mesh_dir meshes/$day \
    -save_args_dir args/$day \
    -save_model_dir model/$day \
    -time_stamp $now \
    -shuffle True \
    -save_plot True \
    -save_model True \
    -save_visual True \
    -save_losses True \
    -save_mesh True \
    -transform False \
    -load_model False \
    -residual True \
    -progress_bar False \
    -normalize False \