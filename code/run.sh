#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"

python grid_search.py \
    -loss_step 1 \
    -log_step 5 \
    -logger_lvl SUCCESS \
    -instance_id 1 \
    -test_ratio 0.1 \
    -val_ratio 0.1 \
    -normalize False \
    -batch_size 16 \
    -out_feature_dim 11 \
    -epochs 100 \
    -ae_layers 3 \
    -ae_ratio 0.3 \
    -hidden_dim 64 \
    -mpl_layers 3 \
    -num_blocks 2 \
    -latent_dim 256 \
    -ae_pool_strat SAG \
    -pool_strat SAG \
    -mpl_ratio 0.5 \
    -opt adam \
    -lr 1e-5 \
    -loss MSE \
    -weight_decay 0.0005 \
    -load_model True\
    -opt_decay_step 30 \
    -opt_decay_rate 0.1 \
    -opt_scheduler step \
    -opt_restart 10 \
    -num_workers 1 \
    -shuffle True \
    -transform True \
    -transform_p 0.1 \
    -load_model False \
    -model_file model_23_11_07-10.54.pt \
    -save_plot_dir plots \
    -save_plot True \
    -save_model True \
    -save_visual True \
    -save_losses True \
    -save_mesh True \
    -time_stamp $now \
    -progress_bar False
