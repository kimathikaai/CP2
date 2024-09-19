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
# Run the none and random variants for the histopathology and polyp datasets

# HIST: No pretrain
for pretrain_type in RANDOM NONE
do
    for dir in panCK_Epithelium
    do
        for ratio in 0.5
        do
            for seed in 0 1 2
            do
                run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}"
                current_dir=${hist_dir}/${dir}
                echo "Fine-tuning ${run_id}"

                CUDA_VISIBLE_DEVICES=2,3 python finetune.py \
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
                    --learning_rate 0.001 \
                    --data_split_type 'RANDOM' \
                    --batch_size 128 \
                    --img_height 512 \
                    --img_width 512 \
                    --epochs 50
            done
        done
    done
done

for pretrain_type in MOCO BYOL CP2
do
    if [ "$pretrain_type" == "MOCO" ]; then
        pretrain_run_id="240917050408-pretrain-MOCO"
    
    elif [ "$pretrain_type" == "BYOL" ]; then
        pretrain_run_id="240917041631-pretrain-BYOL"

    elif [ "$pretrain_type" == "CP2" ]; then
        pretrain_run_id="240917054311-pretrain-CP2"

    else
        echo "Unknown pretrain type: $pretrain_type"
    fi

    echo "Storing model at: ${checkpoint_path}"

    for dir in panCK_Epithelium
    do
        for ratio in 0.5
        do
            for seed in 0 1 2
            do
                run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}"
                current_dir=${hist_dir}/${dir}
                echo "Fine-tuning ${run_id}"

                CUDA_VISIBLE_DEVICES=2,3 python finetune.py \
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
                    --learning_rate 0.001 \
                    --data_split_type 'RANDOM' \
                    --batch_size 64 \
                    --img_height 512 \
                    --img_width 512 \
                    --epochs 50
            done
        done
    done
done
