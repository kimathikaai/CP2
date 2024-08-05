#!/bin/bash
base_dir_0='/mnt/pub0'
base_dir_1='/mnt/pub1'
log_dir="${base_dir_1}/cim-pretraining/logs"
data_dir="${base_dir_1}/ssl-pretraining/data"
pretrain_type=CIM 

# hyper-parameter
num_gpus=2


pretrain_run_id=240803223358-pretrain-CIM
config_file='configs/config_finetune_CIM2.py'
echo "Started fine-tuning for ${pretrain_run_id}"
# Kvasir-SEG CVC-ColonDB ETIS-LaribPolypDB
# 0.3
#  1 2
# --pretrain_path $log_dir/$pretrain_run_id/checkpoint-299.pth \
for dir in CVC-ClinicDB 
	do
		for ratio in 1.0
		do
			for seed in 0
			do
				run_id=$(date +"%y%m%d%H%M%S")-$dir-$pretrain_type-$ratio
				current_dir=${data_dir}/${dir}
				CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
					--pretrain_path $log_dir/$pretrain_run_id/checkpoint-299.pth \
					--pretrain_type $pretrain_type \
					--config $config_file \
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
					--epochs 300
			done
		done
	done
done
