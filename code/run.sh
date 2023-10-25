#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"

python argtest.py \
	-data_dir data/cylinder_flow/  \
    -instance_id 1 \
    -test_ratio 0.1 \
    -val_ratio 0.1 \
    -normalize False \
    -batch_size 16 \
    -out_feature_dim 11 \
    -epochs 100 \
    -ae_layers 2 \
    -ae_ratio 0.5 \
    -hidden_dim 64 \
    -mpl_layers 2 \
    -num_blocks 2 \
    -latent_dim None \
    -pool_strat ASA \
    -mpl_ratio 0.5 \
    -opt adam \
    -lr 0.001 \
    -weight_decay 0.0005 \
    -opt_decay_step 30 \
    -opt_decay_rate 0.1 \
    -opt_scheduler step \
    -opt_restart 10 \
    -num_workers 1 \
    -shuffle True \
    -transform None \
    -transform_p 0.1 \
    -save_model_dir model_chkpoints \
    -save_plot_dir plots \
    -save_plot True \
    -save_args_dir args_chkpoints \
    -save_visualize_dir visualizations \
    -save_mesh_dir meshes \
    -save_model True \
    -save_visual True \
    -save_losses True \
    -save_mesh True \
    -logger_lvl DEBUG \
    -num_layers 2 \
    -time_stamp $now