#!/bin/bash

PROJ_ROOT="."
PROJ_NAME="S-1_5-T_7_10"

python3 ${PROJ_ROOT}/trainer/train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset VIS_dataset \
    --src_address ./data/VIS_dataset/source_train_list.txt \
    --tgt_address ./data/VIS_dataset/target_train_list.txt \
