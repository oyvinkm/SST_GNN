#!/bin/bash

python run.py \
	--datadir 'data/cylinder_flow/'
    --instance_id 1
    --normalize 'False'
    --test_ratio 0.1
    --val_ratio 0.1
    --batch_size 16
    --shuffle 'True'
    --num_workers 1
    --transforms 'None'
    --out_feature_dim 11
    --ae_layers 2
    --ae_ratio 0.5
    --in_dim_node 11
    --hidden_dim 64
    --in_dim_edge 3
    --mpl_layers 2
    --num_blocks 2
    --latent_dim 'None'
    --pool_strat 'ASA'
    --mpl_ratio 0.5
    --opt 'adam'
    --learning_rate 0.001
    --weight_decay 0.0005
    --opt_decay_step 30
    --
    --model_type 'autoencoder' \
    --num_layers 1 \
    --batch_size 16 \
    --hidden_dim 64 \
    