#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"

python run.py \
  	-data_dir data/cylinder_flow/  \
    -loss_step 1 \
    -log_step 1 \
    -logger_lvl INFO \
    -instance_id 1 \
    -test_ratio 0.1 \
    -val_ratio 0.1 \
    -normalize False \
    -batch_size 16 \
    -out_feature_dim 11 \
    -epochs 100 \
    -ae_layers 4 \
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
    -loss None \
    -weight_decay 0.0005 \
    -load_model True\
    -opt_decay_step 30 \
    -opt_decay_rate 0.1 \
    -opt_scheduler step \
    -opt_restart 10 \
    -num_workers 1 \
    -shuffle True \
    -transform None \
    -transform_p 0.2 \
    -load_model True \
    -model_file model_23_11_07-10.54.pt \
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
    -num_layers 2 \
    -time_stamp $now \
    -progress_bar True 
