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
# for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
for dir in CVC-ClinicDB
do
    for ratio in 1
    do
        for seed in 0 1 2
        do
            run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-POLYP"
            current_dir=${data_dir}/${dir}
            echo "Fine-tuning ${run_id}"

            CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
                --pretrain_type $pretrain_type \
                --config $finetune_config_file \
                --seed $seed\
                --run_id $run_id\
                --tags $tags \
                --linear_evaluation \
                --log_dir $log_dir\
                --img_dirs $current_dir/Images \
                --mask_dirs $current_dir/SegmentationImages \
                --train_data_ratio $ratio \
                --learning_rate 0.0001 \
                --num_gpus $num_gpus \
                --num_workers 32 \
                --batch_size 16 \
                --img_height 352 \
                --img_width 352 \
                --epochs 100
        done
    done
done
