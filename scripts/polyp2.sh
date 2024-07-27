#!/bin/bash
base_dir_1='/fast_scratch_2/mahip/Correlational-Image-Modeling'
log_dir="${base_dir_1}/cim-pretraining/logs"
data_dir="${base_dir_1}/data"
pretrain_type=CIM 

# hyper-parameter
num_gpus=2


pretrain_run_id=1236
config_file='configs/config_finetune.py'
echo "Started fine-tuning for ${pretrain_run_id}"
for dir in CVC-ClinicDB Kvasir-SEG CVC-ColonDB ETIS-LaribPolypDB
	do
		for ratio in 1.0 0.3
		do
			for seed in 0 1 2
			do
				run_id=$(date +"%y%m%d%H%M%S")-$dir-$pretrain_type-$ratio
				current_dir=${data_dir}/${dir}
				CUDA_VISIBLE_DEVICES=2,3 python finetune.py \
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
					--img_height 352 \
					--img_width 352 \
					--epochs 100
			done
		done
	done
done
