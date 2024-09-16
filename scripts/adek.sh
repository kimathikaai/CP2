
base_dir_0='/home/kkaai'
# base_dir_1='/mnt/pub1'
log_dir="${base_dir_0}/scratch/log"
data_dir="${base_dir_0}/scratch/data"

# Hyper-parameter
num_gpus=2
finetune_config_file='configs/config_finetune.py'
pretrain_config_file='configs/config_pretrain.py'
model_dir="${data_dir}/models"
# MOCO PIXPRO DINO DENSECL VICEREGL BARLOWTWINS
for pretrain_type in CP2 
do
    # Store models in the specified directory
    model_name="${pretrain_type}"
   
    # Check pretrain_type and set pretrain_run_id accordingly
    if [ "$pretrain_type" == "CP2" ]; then
        checkpoint_path="${model_dir}/${model_name}/cp2-r50-aspp-200ep.pth" # Change this name
    
    elif [ "$pretrain_type" == "MOCO" ]; then
       checkpoint_path="${model_dir}/MOCO-v2/moco_v2_200ep_pretrain.pth.tar" # Change this name
    elif [ "$pretrain_type" == "BYOL" ]; then
        checkpoint_path="${model_dir}/${model_name}/pretrain_res50x1.pkl" # Change this name

    elif [ "$pretrain_type" == "PIXPRO" ]; then
       checkpoint_path="${model_dir}/${model_name}/pixpro_base_r50_400ep_md5_919c6612.pth" # Change this name

    elif [ "$pretrain_type" == "MOCO-v3" ]; then
        checkpoint_path="${model_dir}/${model_name}/checkpoint.ckpt" # Change this name

    elif [ "$pretrain_type" == "DINO" ]; then
        checkpoint_path="${model_dir}/${model_name}/dino_resnet50_pretrain.pth" # Change this name
    elif [ "$pretrain_type" == "DENSECL" ]; then
        checkpoint_path="${model_dir}/${model_name}/densecl_r50_imagenet_200ep.pth" # Change this name

    elif [ "$pretrain_type" == "VICEREGL" ]; then
        checkpoint_path="${model_dir}/${model_name}/resnet50_alpha0.9.pth" # Change this name

    elif [ "$pretrain_type" == "BARLOWTWINS" ]; then
        checkpoint_path="${model_dir}/${model_name}/resnet50.pth" # Change this name
    else
        echo "Unknown pretrain type: $pretrain_type"
    fi

    echo "Storing model at: ${checkpoint_path}"

    for dir in CityScapes
    do 
        for ratio in 0.3 0.6 1.0
        do 
            for seed in 0 1 2
            do
                run_id="$(date +"%y%m%d%H%M%S")-${dir}-${pretrain_type}-R${ratio}-S${seed}-IMAGENET"
                current_dir=${data_dir}/${dir}
                echo "Fine-tuning ${run_id}"

                CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
                    --pretrain_path "${checkpoint_path}" \
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
                    --batch_size 8 \
                    --img_height 352 \
                    --img_width 352 \
                    --epochs 100 --linear_evaluation --num_classes 19
            done
        done
    done
done
