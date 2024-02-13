#!/bin/bash
# strings dont have to be encapsulated in ''
now="$(date +"%y_%m_%d-%H.%M")"

python DirectionsRun.py \
    -epochs 2 \
    -logger_lvl DEBUG \
    -time_stamp $now \