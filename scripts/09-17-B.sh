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

weights_dir='/mnt/pub1/ssl-pretraining/weights'

# Run tests
python -m unittest discover -s tests  -v
tags="$(date +%m-%d-%H%M%S)"


for pretrain_type in DENSECL_IMGNET DINO_IMGNET BARLOWTWINS_IMGNET VICEREGL_IMGNET MOCO_IMGNET PIXPRO_IMGNET BYOL_IMGNET CP2_IMGNET MOSREP_IMGNET CLOVE_IMGNET
do
    # Store models in the specified directory
    model_name="${pretrain_type}"   

    # Check pretrain_type and set pretrain_run_id accordingly
    if [ "$pretrain_type" == "DENSECL_IMGNET" ]; then
        checkpoint_path="${weights_dir}/densecl_r50_imagenet_200ep.pth"
    
    elif [ "$pretrain_type" == "DINO_IMGNET" ]; then
        checkpoint_path="${weights_dir}/dino_resnet50_pretrain.pth"

    elif [ "$pretrain_type" == "BARLOWTWINS_IMGNET" ]; then
        checkpoint_path="${weights_dir}/barlowtwins_resnet50.pth"

    elif [ "$pretrain_type" == "VICEREGL_IMGNET" ]; then
        checkpoint_path="${weights_dir}/viceregl_resnet50_alpha0.9.pth"

    elif [ "$pretrain_type" == "MOCO_IMGNET" ]; then
        checkpoint_path="${weights_dir}/moco_v2_200ep_pretrain.pth.tar"

    elif [ "$pretrain_type" == "MOSREP_IMGNET" ]; then
        checkpoint_path="${weights_dir}/mosaic_in1k_200ep.pth"

    elif [ "$pretrain_type" == "PIXPRO_IMGNET" ]; then
        checkpoint_path="${weights_dir}/pixpro_base_r50_400ep_md5_919c6612.pth"

    elif [ "$pretrain_type" == "CP2_IMGNET" ]; then
        checkpoint_path="${weights_dir}/cp2-r50-aspp-200ep.pth"

    elif [ "$pretrain_type" == "BYOL_IMGNET" ]; then
        checkpoint_path="${weights_dir}/byol_resnet50x1.pth.tar"

    elif [ "$pretrain_type" == "CLOVE_IMGNET" ]; then
        checkpoint_path="${weights_dir}/clove_200ep_mc.pth"

    else
        echo "Unknown pretrain type: $pretrain_type"
    fi

    echo "Storing model at: ${checkpoint_path}"

    for dir in glas
    do
        for ratio in 1
        do
            for seed in 0 1 2
            do
                run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-IMGNET"
                current_dir=${hist_dir}/${dir}
                echo "Fine-tuning ${run_id}"

                CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
                    --pretrain_path $checkpoint_path \
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
    for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
    do
        for ratio in 1
        do
            for seed in 0 1 2
            do
                run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-IMGNET"
                current_dir=${data_dir}/${dir}
                echo "Fine-tuning ${run_id}"

                CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
                    --pretrain_path $checkpoint_path \
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
