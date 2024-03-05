#!/bin/bash
# strings dont have to be encapsulated in ''

now="$(date +"%y_%m_%d-%H.%M")"
day="$(date +"%d-%m-%y")"
target="19-02-24/test2"         # used when we want to load a model
# Some warningsremoval for floating-point round-off errors
prefix="../logs"
TF_ENABLE_ONEDNN_OPTS=0 python VGAErun.py \
    -ae_layers 6 \
    -data_augmentation True \
    -batch_size 2 \
    -epochs 35 \
    -instance_id 1 \
    -loss_step 10 \
    -log_step 5 \
    -lr 1e-4 \
    -latent_dim 512 \
    -logger_lvl info \
    -loss LMSE \
    -load_model False \
    -mpl_layers 1 \
    -make_gif True \
    -model_file model.pt \
    -num_blocks 1 \
    -normalize false \
    -out_feature_dim 11 \
    -random_search True \
    -save_args_dir $prefix/args/$day \
    -save_encodings False \
    -save_gif_dir $prefix/gifs/$day \
    -save_losses true \
    -save_mesh true \
    -save_mesh_dir $prefix/meshes/$day \
    -save_model true \
    -save_model_dir $prefix/model_chkpoints/$day \
    -save_plot_dir $prefix/plots/$day \
    -save_plot true \
    -save_visual true \
    -test_ratio 0.1 \
    -time_stamp $now \
    -train True \
    -pretext_task False \
    -transform_p 0.3 \
    -val_ratio 0.1 \
