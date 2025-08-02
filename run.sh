#!/bin/bash

run_evaluation() {
    ARRANGEMENT=$1
    CONDITION=$2
    SAM2_CFG=$3
    SAM2_CHECKPOINT=$4

    CONFIG_NAME=$(basename "$SAM2_CFG" .yaml)

    python /home/user/TFSeg/main.py \
        --sam2_cfg "$SAM2_CFG" \
        --sam2_checkpoint "$SAM2_CHECKPOINT" \
        --train_dir './data/BUS512_tvt/train' \
        --val_dir './data/BUS512_tvt/val' \
        --test_dir './data/BUS512_tvt/test' \
        --result_save_dir 'output' \
        --arrange "$ARRANGEMENT" \
        --add_condition "$CONDITION"
}

run_evaluation "descend" "mask" "sam2_hiera_l.yaml" "../sam2/checkpoints/sam2_hiera_large.pt"
