#!/bin/bash
base_dir_0='/mnt/pub0'
base_dir_1='/mnt/pub1'
log_dir="${base_dir_1}/ssl-pretraining/logs"
data_dir="${base_dir_1}/ssl-pretraining/data"

# Hyper-parameter
num_gpus=2
finetune_config_file='configs/config_finetune.py'
pretrain_config_file='configs/config_pretrain.py'

# Run tests
python -m unittest discover -s tests  -v

#
# NO PRE-TRAINING
#

pretrain_type='NONE'
for dir in glas_dataset
do
    for ratio in 0.3 1.0
    do
        for seed in 0 1 2
        do
            run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}"
            current_dir=${data_dir}/${dir}
            echo "Fine-tuning ${run_id}"

            CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
                --pretrain_type $pretrain_type \
                --config $finetune_config_file \
                --seed $seed\
                --run_id $run_id\
                --log_dir $log_dir\
                --img_dirs $current_dir/Images \
                --mask_dirs $current_dir/SegmentationImages \
                --train_data_ratio $ratio \
                --num_gpus $num_gpus \
                --num_workers 32 \
                --batch_size 16 \
                --img_height 512 \
                --img_width 512 \
                --epochs 100
        done
    done
done

#
# HIST PRE-TRAINING
#

for pretrain_type in CP2 MOCO BYOL
do
    #
    # PRE-TRAINING
    #

    # Create logging name
    pretrain_run_id="$(date +"%y%m%d%H%M%S")-pretrain-${pretrain_type}-HIST"
    echo "Started pre-training for ${pretrain_run_id}"

    # Start pre-training
    CUDA_VISIBLE_DEVICES=0,1 python main.py \
        --seed 0 \
        --run_id $pretrain_run_id \
        --log_dir $log_dir \
        --pretrain_type $pretrain_type \
        --data_dirs "$data_dir/glas_dataset/Images" "$data_dir/kumar/Images" "$data_dir/cpm15/Images" "$data_dir/cpm17/Images" "$data_dir/tnbc/Images" "$data_dir/correctoral_cancer_dataset/Images" \
        --config $pretrain_config_file \
        --epochs 200 \
        --lr 0.001 \
        --num-workers 64 \
        --batch-size 32 \
        --world-size $num_gpus \
        --foreground_min 0.5 \
        --foreground_max 0.8 \
        --backbone_type 'DEEPLABV3' \
        --cap_queue

    #
    # FINE-TUNE
    #

    for dir in glas_dataset
    do
        for ratio in 0.3 1.0
        do
            for seed in 0 1 2
            do
                run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-HIST"
                current_dir=${data_dir}/${dir}
                echo "Fine-tuning ${run_id}"

                CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
                    --pretrain_path "${log_dir}/${pretrain_run_id}/checkpoint.ckpt" \
                    --pretrain_type $pretrain_type \
                    --config $finetune_config_file \
                    --seed $seed\
                    --run_id $run_id\
                    --log_dir $log_dir\
                    --img_dirs $current_dir/Images \
                    --mask_dirs $current_dir/SegmentationImages \
                    --train_data_ratio $ratio \
                    --num_gpus $num_gpus \
                    --num_workers 32 \
                    --batch_size 16 \
                    --img_height 512 \
                    --img_width 512 \
                    --epochs 100
            done
        done
    done
done

#
# IMAGENET PRE-TRAINING
#

for pretrain_type in CP2 MOCO BYOL
do
    #
    # PRE-TRAINING
    #

    # Create logging name
    pretrain_run_id="$(date +"%y%m%d%H%M%S")-pretrain-${pretrain_type}-IMAGENET"
    echo "Started pre-training for ${pretrain_run_id} IMAGENET"

    # Start pre-training
    CUDA_VISIBLE_DEVICES=0,1 python main.py \
        --seed 0 \
        --run_id $pretrain_run_id \
        --log_dir $log_dir \
        --pretrain_type $pretrain_type \
        --data_dirs "${data_dir}/imagenet1k0_70194/Images" \
        --config $pretrain_config_file \
        --epochs 200 \
        --lr 0.001 \
        --num-workers 64 \
        --batch-size 32 \
        --world-size $num_gpus \
        --foreground_min 0.5 \
        --foreground_max 0.8 \
        --backbone_type 'DEEPLABV3' \
        --cap_queue

    #
    # FINE-TUNE
    #

    for dir in glas_dataset
    do
        for ratio in 0.3 1.0
        do
            for seed in 0 1 2
            do
                run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-IMAGENET"
                current_dir=${data_dir}/${dir}
                echo "Fine-tuning ${run_id}"

                CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
                    --pretrain_path "${log_dir}/${pretrain_run_id}/checkpoint.ckpt" \
                    --pretrain_type $pretrain_type \
                    --config $finetune_config_file \
                    --seed $seed\
                    --run_id $run_id\
                    --log_dir $log_dir\
                    --img_dirs $current_dir/Images \
                    --mask_dirs $current_dir/SegmentationImages \
                    --train_data_ratio $ratio \
                    --num_gpus $num_gpus \
                    --num_workers 32 \
                    --batch_size 16 \
                    --img_height 512 \
                    --img_width 512 \
                    --epochs 100
            done
        done
    done
done
