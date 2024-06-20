#!/bin/bash
base_dir_0='/mnt/pub0'
base_dir_1='/mnt/pub1'
log_dir="${base_dir_1}/ssl-pretraining/logs"
data_dir="${base_dir_1}/ssl-pretraining/data"

# hyper-parameter
num_gpus=2

# for pretrain_type in MOCO BYOL
for pretrain_type in MIRROR
do
	# Get the correct config file
	# if [[ "$pretrain_type" == "CP2" ]]; then
	# 	pretrain_config_file='configs/config_pretrain.py'
	# else
	# 	pretrain_config_file='configs/config_moco.py'
	# fi
	pretrain_config_file='configs/config_pretrain.py'
	
	# Create logging name
	pretrain_run_id=$(date +"%y%m%d%H%M%S")-pretrain-$pretrain_type
	echo "Started pre-training for ${pretrain_run_id}"
	# Start pre-training
	CUDA_VISIBLE_DEVICES=0,1 python mirror_pretrain.py \
	    --seed 0 \
	    --run_id $pretrain_run_id \
	    --log_dir $log_dir \
	    --data_dirs "$data_dir/CVC-ClinicDB/Images" "$data_dir/CVC-ColonDB/Images" "$data_dir/ETIS-LaribPolypDB/Images" "$data_dir/Kvasir-SEG/Images"   \
	    --config $pretrain_config_file \
	    --epochs 200 \
	    --lr 0.001 \
	    --num-workers 32 \
	    --batch-size 16 \
	    --num_gpus $num_gpus \
        --variant "NONE" \
        --max_num_patches 1 \
        --max_area_scale 0.05 \
        --min_area_scale 0.005 \
        --img_x_size 512 \
        --img_y_size 512

	# Get the correct config file
	# if [[ "$pretrain_type" == "CP2" ]]; then
	# 	config_file='configs/config_finetune.py'
	# else
	# 	config_file='configs/config_finetune_moco.py'
	# fi
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
				CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
					--pretrain_path $log_dir/$pretrain_run_id/checkpoint.ckpt \
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
