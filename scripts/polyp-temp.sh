#!/bin/bash
base_dir_0='/mnt/pub0'
base_dir_1='/mnt/pub1'
log_dir="${base_dir_1}/ssl-pretraining/logs"
data_dir="${base_dir_1}/ssl-pretraining/data"
pretrain_dir='/mnt/pub1/ssl-pretraining/data/hyper-kvasir/unlabeled-images/images'

# Hyper-parameter
num_gpus=2
finetune_config_file='configs/config_finetune.py'
pretrain_config_file='configs/config_pretrain.py'

# Run tests
python -m unittest discover -s tests  -v
tags="$(date +%m-%d-%H%M%S)-ablation"

pretrain_type=PROPOSED
for pretrain_run_id in '240909211944-pretrain-PROPOSED-POLYP-ABLATION' '240909202835-pretrain-PROPOSED-POLYP-ABLATION' '240909193747-pretrain-PROPOSED-POLYP-ABLATION' '240909184720-pretrain-PROPOSED-POLYP-ABLATION' '240909175630-pretrain-PROPOSED-POLYP-ABLATION' '240909170519-pretrain-PROPOSED-POLYP-ABLATION' '240909161408-pretrain-PROPOSED-POLYP-ABLATION' '240909152259-pretrain-PROPOSED-POLYP-ABLATION' '240909143148-pretrain-PROPOSED-POLYP-ABLATION' '240909134020-pretrain-PROPOSED-POLYP-ABLATION' '240909123815-pretrain-PROPOSED-POLYP-ABLATION' '240909103212-pretrain-PROPOSED-POLYP-ABLATION' '240909082516-pretrain-PROPOSED-POLYP-ABLATION' '240909061733-pretrain-PROPOSED-POLYP-ABLATION' '240909040919-pretrain-PROPOSED-POLYP-ABLATION' '240909020053-pretrain-PROPOSED-POLYP-ABLATION' '240908235155-pretrain-PROPOSED-POLYP-ABLATION' '240908214344-pretrain-PROPOSED-POLYP-ABLATION' '240908193741-pretrain-PROPOSED-POLYP-ABLATION' '240908173249-pretrain-PROPOSED-POLYP-ABLATION' '240907084756-pretrain-PROPOSED-POLYP-ABLATION'
do
    for dir in Kvasir-SEG CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB
    # for dir in Kvasir-SEG CVC-ClinicDB
    do
        for ratio in 1.0
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
                    --offline_wandb \
                    --log_dir $log_dir\
                    --img_dirs $current_dir/Images \
                    --mask_dirs $current_dir/SegmentationImages \
                    --train_data_ratio $ratio \
                    --learning_rate 0.0001 \
                    --num_gpus $num_gpus \
                    --linear_evaluation \
                    --num_workers 32 \
                    --batch_size 16 \
                    --img_height 352 \
                    --img_width 352 \
                    --epochs 100
            done
        done
    done
done

