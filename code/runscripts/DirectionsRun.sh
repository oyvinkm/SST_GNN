#!/bin/bash
# strings dont have to be encapsulated in ''
now="$(date +"%y_%m_%d-%H.%M")"
python DirectionsRun.py \
    -decoder_path ../logs/model_chkpoints/decoder.pt \
    -decode_test False \
    -device cuda \
    -epochs 2 \
    -logger_lvl DEBUG \
    -make_gif False \
    # -time_stamp $now \
