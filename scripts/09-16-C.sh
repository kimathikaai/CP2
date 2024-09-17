#!/bin/bash
base_dir_0='/mnt/pub0'
base_dir_1='/mnt/pub1'
log_dir="${base_dir_1}/ssl-pretraining/logs"
data_dir="${base_dir_1}/ssl-pretraining/data"
poly_pretrain_dir='/mnt/pub1/ssl-pretraining/data/hyper-kvasir/unlabeled-images/images'
hist_dir='/mnt/pub1/ssl-pretraining/data/histopathology'
hist_pretrain_dir="${hist_dir}/aSMA_SmoothMuscle/Images"

# Hyper-parameter
num_gpus=2
finetune_config_file='configs/config_finetune.py'
pretrain_config_file='configs/config_pretrain.py'

# Run tests
python -m unittest discover -s tests  -v
tags="$(date +%m-%d-%H%M%S)"

# DESCRIPTION
# Histopathology 'glas' dataset performance pre-trained on the large SegPath-aSMA dataset

# No pretrain
for pretrain_type in NONE RANDOM
do
    for dir in glas
    do
        for ratio in 1
        do
            for seed in 0 1 2
            do
                run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}"
                current_dir=${hist_dir}/${dir}
                echo "Fine-tuning ${run_id}"

                CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
                    --pretrain_type $pretrain_type \
                    --config $finetune_config_file \
                    --seed $seed\
                    --run_id $run_id\
                    --tags $tags \
                    --log_dir $log_dir\
                    --img_dirs $current_dir/Images \
                    --mask_dirs $current_dir/SegmentationImages \
                    --train_data_ratio $ratio \
                    --num_gpus $num_gpus \
                    --num_workers 32 \
                    --batch_size 16 \
                    --img_height 352 \
                    --img_width 352 \
                    --offline_wandb \
                    --epochs 100
            done
        done
    done
done

#
# PixPro
#
for pretrain_type in BYOL MOCO CP2
do
    #
    # PRE-TRAINING
    #

    # Create logging name
    pretrain_run_id="$(date +"%y%m%d%H%M%S")-pretrain-${pretrain_type}"
    echo "Started pre-training for ${pretrain_run_id}"

    # Start pre-training
    CUDA_VISIBLE_DEVICES=0,1 python main.py \
        --seed 0 \
        --run_id $pretrain_run_id \
        --log_dir $log_dir \
        --tags $tags \
        --pretrain_type $pretrain_type \
        --data_dirs $hist_pretrain_dir \
        --config $pretrain_config_file \
        --epochs 10 \
        --lr 0.001 \
        --num-workers 64 \
        --batch-size 128 \
        --world-size $num_gpus \
        --foreground_min 0.5 \
        --foreground_max 0.8 \
        --backbone_type 'DEEPLABV3' \
        --cap_queue

    #
    # FINE-TUNE
    #
    for dir in glas
    do
        for ratio in 1
        do
            for seed in 0 1 2
            do
                run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}"
                current_dir=${hist_dir}/${dir}
                echo "Fine-tuning ${run_id}"

                CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
                    --pretrain_path "${log_dir}/${pretrain_run_id}/checkpoint.ckpt" \
                    --pretrain_type $pretrain_type \
                    --config $finetune_config_file \
                    --seed $seed\
                    --run_id $run_id\
                    --tags $tags \
                    --log_dir $log_dir\
                    --img_dirs $current_dir/Images \
                    --mask_dirs $current_dir/SegmentationImages \
                    --train_data_ratio $ratio \
                    --num_gpus $num_gpus \
                    --num_workers 32 \
                    --batch_size 16 \
                    --img_height 352 \
                    --img_width 352 \
                    --offline_wandb \
                    --epochs 100
            done
        done
    done
done

#
# PixPro
#
