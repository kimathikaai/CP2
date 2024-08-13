#!/bin/bash
base_dir_0='/mnt/pub0'
base_dir_1='/mnt/pub1'
log_dir="${base_dir_1}/ssl-pretraining/logs"
data_dir="${base_dir_1}/ssl-pretraining/data"

# hyper-parameter
num_gpus=2

# for pretrain_type in MOCO BYOL
for pretrain_type in CP2
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
	CUDA_VISIBLE_DEVICES=1,2 python main.py \
	    --seed 0 \
	    --run_id $pretrain_run_id \
	    --log_dir $log_dir \
	    --pretrain_type $pretrain_type \
        --data_dirs "$base_dir_0/data/CGS_semseg_iphone_2018_resized/Images" "$base_dir_0/data/CGS_semseg_iphone_2019_resized/Images" \
	    --config $pretrain_config_file \
	    --epochs 200 \
	    --lr 0.001 \
	    --num-workers 64 \
	    --batch-size 32 \
	    --world-size $num_gpus \
	    --foreground_min 0.25 \
	    --foreground_max 0.5 \
        --backbone_type 'UNET_ENCODER_ONLY' \
        --mapping_type 'PIXEL_PIXEL_REGION' \
        --lmbd_corr_weight 10 \
	    --cap_queue \
        --lemon_data
done
