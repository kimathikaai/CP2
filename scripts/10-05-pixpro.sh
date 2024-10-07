#!/bin/bash
base_dir_0='/mnt/pub0'
base_dir_1='/mnt/pub1'
log_dir="${base_dir_1}/ssl-pretraining/logs"
data_dir="${base_dir_1}/ssl-pretraining/data"
poly_pretrain_dir='/mnt/pub1/ssl-pretraining/data/hyper-kvasir/unlabeled-images/images'
hist_dir='/mnt/pub1/ssl-pretraining/data/histopathology'
hist_pretrain_dir="${hist_dir}/aSMA_SmoothMuscle/Images"
hyperkvasir_segpathasma='/mnt/pub1/ssl-pretraining/data/hyperkvasir-segpathasma'

# Hyper-parameter
num_gpus=2
finetune_config_file='configs/config_finetune.py'
pretrain_config_file='configs/config_pretrain.py'

#
# PIXPRO Pre-training
#
set -x

cd /home/kkaai/workspace/PixPro
deactivate
source /home/kkaai/envs/pix/bin/activate
run_id="$(date +"%y%m%d%H%M%S")-PixPro-PHHS"
# --zip  
# python main_pretrain.py \
# --amp-opt-level O1 \
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 12348 --nproc_per_node=2 \
    main_pretrain.py \
    --seed 0 \
    --run_id $run_id \
    --data-dir ${hyperkvasir_segpathasma} \
    --output-dir "${base_dir_1}/ssl-pretraining/logs/${run_id}" \
    \
    --cache-mode no \
    --crop 0.08 \
    --aug BYOL \
    --dataset ImageNet \
    --batch-size 128 \
    --num-workers 32 \
    \
    --model PixPro \
    --arch resnet50 \
    --head-type early_return \
    \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 10 \
    --amp-opt-level O0 \
    \
    --save-freq 10 \
    --auto-resume \
    \
    --pixpro-p 2 \
    --pixpro-momentum 0.99 \
    --pixpro-pos-ratio 0.7 \
    --pixpro-transform-layer 1 \
    --pixpro-ins-loss-weight 1 \

deactivate
source /home/kkaai/envs/cp2/bin/activate
cd /home/kkaai/workspace/CP2
git pull origin
git checkout densecl

# Run tests
python -m unittest discover -s tests  -v
tags="$(date +%m-%d-%H%M%S)"

pretrain_type=PIXPRO
pretrain_run_id=$run_id

#
# FINETUNE - POLYP
#
for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
do
    for ratio in 0.3 1
    do
        for seed in 0 1 2
        do
            run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-PA"
            current_dir=${data_dir}/${dir}
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
                --batch_size 16 \
                --img_height 352 \
                --img_width 352 \
                --epochs 100
        done
    done
done
#
# FINETUNE - HIST
# 
for dir in glas
do
    for ratio in 1
    do
        for seed in 0 1 2
        do
            run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-H"
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
                --batch_size 16 \
                --img_height 352 \
                --img_width 352 \
                --epochs 100
        done
    done
done
