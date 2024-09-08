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
tags="$(date +%m-%d-%H%M%S)"


pretrain_type='NONE'
for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
do
    for ratio in 0.3 0.6
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

pretrain_type='BYOL'
pretrain_path='/mnt/pub1/ssl-pretraining/logs/240816170945-pretrain-BYOL-POLYP/checkpoint.ckpt'
for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
do
    for ratio in 0.3 0.6
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

pretrain_type='CP2'
pretrain_path='/mnt/pub1/ssl-pretraining/logs/240816040556-pretrain-CP2-POLYP/checkpoint.ckpt'
for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
do
    for ratio in 0.3 0.6
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

pretrain_type='MOCO'
pretrain_path='/mnt/pub1/ssl-pretraining/logs/240816115548-pretrain-MOCO-POLYP/checkpoint.ckpt'
for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
do
    for ratio in 0.3 0.6
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

pretrain_type='PIXPRO'
pretrain_path='/mnt/pub1/ssl-pretraining/logs/240902172247-PixPro/checkpoint.ckpt'
for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
do
    for ratio in 0.3 0.6
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

pretrain_type='PROPOSED'
pretrain_path='/mnt/pub1/ssl-pretraining/logs/240905185852-pretrain-PROPOSED-POLYP/checkpoint.ckpt'
for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
do
    for ratio in 0.3 0.6
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

pretrain_type=PROPOSED
for negative_type in FIXED AVERAGE NONE
do
    for lmbd_pixel_corr_weight in 10 1
    do
        for lmbd_region_corr_weight in 0 1
        do
            for lmbd_not_corr_weight in 0 1
            do
                for pixel_ids_stride in 1
                do
                    #
                    # PRE-TRAINING
                    #

                    # Create logging name
                    pretrain_run_id="$(date +"%y%m%d%H%M%S")-pretrain-${pretrain_type}-POLYP-ABLATION"
                    echo "Started pre-training for ${pretrain_run_id}"

                    # Start pre-training
                    CUDA_VISIBLE_DEVICES=0,1 python main.py \
                    --seed 0 \
                    --run_id $pretrain_run_id \
                    --tags $tags \
                    --log_dir $log_dir \
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
                    --mapping_type 'PIXEL_REGION_ID' \
                    --negative_type $negative_type \
                    --lmbd_pixel_corr_weight $lmbd_pixel_corr_weight\
                    --lmbd_region_corr_weight $lmbd_region_corr_weight\
                    --lmbd_not_corr_weight $lmbd_not_corr_weight\
                    --pixel_ids_stride $pixel_ids_stride \
                    --cap_queue
                    #
                    # FINE-TUNE
                    #

                    for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
                    # for dir in Kvasir-SEG CVC-ClinicDB
                    do
                        for ratio in 0.3 0.6 1.0
                        do
                            for seed in 0 1 2
                            do
                                run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-POLYP-ABLATION"
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
                done
            done
        done
    done
done
