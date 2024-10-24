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


# MOCO BYOL
for pretrain_type in CP2 
do
#
# PRE-TRAIN (HyperK+segpath)
#
    pretrain_run_id="$(date +"%y%m%d%H%M%S")-pretrain-${pretrain_type}-PHHS"
    echo "Started pre-training for ${pretrain_run_id}"
    # Start pre-training
    CUDA_VISIBLE_DEVICES=2,3 python main.py \
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
        --cap_queue 

done
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

            CUDA_VISIBLE_DEVICES=0,2 python finetune.py \
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
# #
# # FINETUNE - HIST
# # 
# for dir in glas
# do
#     for ratio in 1
#     do
#         for seed in 0 1 2
#         do
#             run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-H"
#             current_dir=${hist_dir}/${dir}
#             echo "Fine-tuning ${run_id}"

#             CUDA_VISIBLE_DEVICES=0,2 python finetune.py \
#                 --pretrain_path "${log_dir}/${pretrain_run_id}/checkpoint.ckpt" \
#                 --pretrain_type $pretrain_type \
#                 --config $finetune_config_file \
#                 --seed $seed\
#                 --run_id $run_id\
#                 --tags $tags \
#                 --log_dir $log_dir\
#                 --img_dirs $current_dir/Images \
#                 --mask_dirs $current_dir/SegmentationImages \
#                 --train_data_ratio $ratio \
#                 --num_gpus $num_gpus \
#                 --num_workers 32 \
#                 --batch_size 16 \
#                 --img_height 352 \
#                 --img_width 352 \
#                 --epochs 100
#         done
#     done
# done