#!/bin/bash
base_dir_0='/mnt/pub0'
base_dir_1='/mnt/pub1'
log_dir="${base_dir_1}/ssl-pretraining/logs"
data_dir="${base_dir_1}/ssl-pretraining/data"

num_gpus=2
finetune_config_file='configs/config_finetune.py'

# Run tests
python -m unittest discover -s tests  -v

pretrain_type='PIXPRO'
for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
do
    for ratio in 0.3 1.0
    do
        for seed in 0 1 2
        do
            run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}"
            current_dir=${data_dir}/${dir}
            echo "Fine-tuning ${run_id}"

            CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
                --pretrain_path "/mnt/pub1/ssl-pretraining/logs/240902172247-PixPro/checkpoint.ckpt" \
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
                --img_height 352 \
                --img_width 352 \
                --epochs 100
        done
    done
done
