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
# 1) Evaluate the impact of reducing labelled data has with respect
#    to the ImageNet-Supervised (NONE) baseline
# 2) This is for the HyperKvasir baseline
#

run_type=test
# run_type=go


for ratio in 0.1 0.6 0.8 0.9
do
    for pretrain_type in RANDOM NONE MOCO BYOL CP2 DENSECL PIXPRO PROPOSED_V2
    do
        if [ "$pretrain_type" == "MOCO" ]; then
            pretrain_run_id="240914100741-pretrain-MOCO-POLYP"
        
        elif [ "$pretrain_type" == "PROPOSED_V2" ]; then
            pretrain_run_id="241016144052-pretrain-PROPOSED_V2-PH"

        elif [ "$pretrain_type" == "BYOL" ]; then
            pretrain_run_id="240914015940-pretrain-BYOL-POLYP"

        elif [ "$pretrain_type" == "CP2" ]; then
            pretrain_run_id="240913034138-pretrain-CP2-POLYP"

        elif [ "$pretrain_type" == "DENSECL" ]; then
            pretrain_run_id="241012064808-pretrain-DENSECL-PH"

        elif [ "$pretrain_type" == "PIXPRO" ]; then
            pretrain_run_id="240917160051-PixPro"

        elif [ "$pretrain_type" == "NONE" ]; then
            pretrain_run_id="x"

        elif [ "$pretrain_type" == "RANDOM" ]; then
            pretrain_run_id="x"

        else
            echo "Unknown pretrain type: $pretrain_type"
        fi

        if [ "$run_type" == "test" ]; then
            for dir in ETIS-LaribPolypDB
            do
                for seed in 0
                do
                    run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-PHPA"
                    current_dir=${data_dir}/${dir}
                    echo "Fine-tuning ${run_id}"

                    CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
                        --fast_dev_run \
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

        elif [ "$run_type" == "go" ]; then
            for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
            do
                for seed in 0 1 2
                do
                    run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-PHPA"
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
        else
            echo "Unknown run_type: $run_type"
        fi
    done
done
