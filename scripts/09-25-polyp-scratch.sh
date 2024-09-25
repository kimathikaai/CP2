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
# 1) Needed to see how well these methods would do when training from scratch
# 2) Polyp segmentation with only 10% of the data available. Aggregate weights

# Polyp aggregate weights
for pretrain_type in RANDOM NONE MOCO BYOL CP2
do
    if [ "$pretrain_type" == "MOCO" ]; then
        pretrain_run_id="240816115548-pretrain-MOCO-POLYP"
        pretrain_path="${log_dir}/${pretrain_run_id}/checkpoint.ckpt"
    
    elif [ "$pretrain_type" == "BYOL" ]; then
        pretrain_run_id="240816170945-pretrain-BYOL-POLYP"
        pretrain_path="${log_dir}/${pretrain_run_id}/checkpoint.ckpt"

    elif [ "$pretrain_type" == "CP2" ]; then
        pretrain_run_id="240816040556-pretrain-CP2-POLY"
        pretrain_path="${log_dir}/${pretrain_run_id}/checkpoint.ckpt"

    elif [ "$pretrain_type" == "NONE" ]; then
        pretrain_path="x"

    elif [ "$pretrain_type" == "RANDOM" ]; then
        pretrain_path="x"

    else
        echo "Unknown pretrain type: $pretrain_type"
    fi

    echo "Storing model at: ${checkpoint_path}"

    for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
    do
        for ratio in 0.1
        do
            for seed in 0 1 2
            do
                run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-POLYP"
                current_dir=${data_dir}/${dir}
                echo "Fine-tuning ${run_id}"

                CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
                    --pretrain_path $pretrain_path \
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
                    --epochs 100
            done
        done
    done
done

#
# POLYP PRE-TRAINING
#

for pretrain_type in CP2
do
    #
    # PRE-TRAINING
    #

    # Create logging name
    pretrain_run_id="$(date +"%y%m%d%H%M%S")-pretrain-${pretrain_type}-Polyp"
    echo "Started pre-training for ${pretrain_run_id}"

    # Start pre-training
    CUDA_VISIBLE_DEVICES=0,1 python main.py \
        --seed 0 \
        --run_id $pretrain_run_id \
        --log_dir $log_dir \
        --tags $tags \
        --pretrain_type $pretrain_type \
        --data_dirs "$data_dir/CVC-ClinicDB/Images" "$data_dir/CVC-ColonDB/Images" "$data_dir/ETIS-LaribPolypDB/Images" "$data_dir/Kvasir-SEG/Images"   \
        --config $pretrain_config_file \
        --epochs 200 \
        --lr 0.001 \
        --num-workers 64 \
        --batch-size 32 \
        --world-size $num_gpus \
        --foreground_min 0.5 \
        --foreground_max 0.8 \
        --backbone_type 'DEEPLABV3' \
        --pretrain_from_scratch \
        --cap_queue

    #
    # FINE-TUNE
    #

    for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
    do
        for ratio in 0.3
        do
            for seed in 0 1 2
            do
                run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-POLYP"
                current_dir=${data_dir}/${dir}
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
                    --epochs 100
            done
        done
    done
done

