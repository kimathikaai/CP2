#!/bin/bash
base_dir_0='/mnt/pub0'
base_dir_1='/mnt/pub1'
log_dir="${base_dir_1}/ssl-pretraining/logs"
data_dir="${base_dir_1}/ssl-pretraining/data"

# Hyper-parameter
num_gpus=1
finetune_config_file='configs/config_finetune.py'
pretrain_config_file='configs/config_pretrain.py'

# Run tests
python -m unittest discover -s tests  -v
tags="$(date +%m-%d-%H%M%S)-reproducability"


pretrain_type='NONE'
#
# PRE-TRAINING
#

# Create logging name
pretrain_run_id="$(date +"%y%m%d%H%M%S")-REPRODUCE"
echo "Started pre-training for ${pretrain_run_id}"

# Start pre-training
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --seed 0 \
    --run_id $pretrain_run_id \
    --log_dir $log_dir \
    --pretrain_type $pretrain_type \
    --data_dirs $pretrain_dir \
    --config $pretrain_config_file \
    --max_steps 10 \
    --lr 0.001 \
    --num-workers 64 \
    --batch-size 32 \
    --world-size $num_gpus \
    --foreground_min 0.5 \
    --foreground_max 0.8 \
    --backbone_type 'DEEPLABV3' \
    --cap_queue
