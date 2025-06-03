#!/bin/bash

# example usage
# sh scripts/finetune_with_accelerate_config.sh 1 configs/train_configs/sft/default.yaml
# sh scripts/finetune_with_accelerate_config.sh 8 configs/train_configs/sft/olmo_17_sft.yaml

# Check if exactly two arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <num_gpus>"
    echo "Example: $0 2"
    exit 1
fi

NUM_GPUS="$1"

# Generate CUDA_VISIBLE_DEVICES as a range from 0 to NUM_GPUS-1
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES

echo "Number of GPUs: $NUM_GPUS"
# echo "Using config file: $CONFIG_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.
TOTAL_BATCH_SIZE=128

for reduction in sum; # mean;
do
    for learning_rate in 5e-6 2e-5 8e-5; # 2e-4 8e-4; # 2e-3 8e-3;
    do
        for smoothing in 0.0 0.1; # 0.2; #0.3 0.4 0.5; 
        do
            # for config in \
            #     configs/train_configs/tulu3/gemma_sft_2b.yaml \
            #     configs/train_configs/tulu3/gemma2_sft_2b.yaml \
            #     configs/train_configs/tulu3/qwen2_5_sft_0_5b.yaml \
            #     configs/train_configs/tulu3/qwen2_5_sft_1_5b.yaml \
            #     configs/train_configs/tulu3/qwen2_5_sft_3b.yaml \
            #     configs/train_configs/tulu3/qwen2_5_sft_7b.yaml \
            #     configs/train_configs/tulu3/tulu2_sft_7b.yaml \
            #     configs/train_configs/tulu3/tulu3_sft_8b.yaml \
            #     configs/train_configs/tulu3/tulu3_2_sft_1b.yaml \
            #     configs/train_configs/tulu3/tulu3_2_sft_3b.yaml \
            #     configs/train_configs/tulu3/mistral_sft_7b.yaml;
            # ;
            # do
            
            # for config in \
            #     configs/train_configs/tulu3/mistral_sft_7b.yaml \
            #     configs/train_configs/tulu3/tulu3_sft_8b.yaml \
            #     configs/train_configs/tulu3/tulu2_sft_7b.yaml \
            #     configs/train_configs/tulu3/tulu2_sft_13b.yaml \
            #     configs/train_configs/tulu3/gemma_sft_2b.yaml \
            #     configs/train_configs/tulu3/gemma2_sft_2b.yaml \
            #     configs/train_configs/tulu3/gemma_sft_9b.yaml \
            #     configs/train_configs/tulu3/gemma2_sft_9b.yaml \
            #     configs/train_configs/tulu3/tulu3_2_sft_3b.yaml \
            #     configs/train_configs/tulu3/tulu3_2_sft_1b.yaml \
            #     configs/train_configs/tulu3/qwen2_5_sft_0_5b.yaml \
            #     configs/train_configs/tulu3/qwen2_5_sft_1_5b.yaml \
            #     configs/train_configs/tulu3/qwen2_5_sft_3b.yaml \
            #     configs/train_configs/tulu3/qwen2_5_sft_7b.yaml \
            #     configs/train_configs/tulu3/qwen2_5_sft_14b.yaml \
            # ;
            # do

            for temperature in 0.05 0.25 0.5 0.75;
            do
                BATCH_SIZE_PER_GPU=8
                GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
                for config in \
                    configs/train_configs/tulu3/tulu3_2_sft_1b.yaml \
                    configs/train_configs/tulu3/qwen2_5_sft_0_5b.yaml \
                ;
                do
                    NAME=$(echo $config | cut -d'.' -f 1 | rev | cut -d'/' -f 1 | rev)
                    if [ ! -f /work/jerry/output-temperature-alpaca/$temperature/$reduction/$learning_rate/$NAME-$smoothing/config.json ]; then
                        accelerate launch \
                        --mixed_precision bf16 \
                        --main_process_port 29600 \
                        --num_machines 1 \
                        --num_processes $NUM_GPUS \
                        --use_deepspeed \
                        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
                        open_instruct/finetune-working.py \
                        $config \
                        --temperature=$temperature \
                        --per_device_train_batch_size=$BATCH_SIZE_PER_GPU \
                        --gradient_accumulation_steps=$GRADIENT_ACC_STEPS \
                        --label_smoothing=$smoothing \
                        --learning_rate=$learning_rate \
                        --run_name=$NAME-$smoothing \
                        --output_dir=/work/jerry/output-temperature-alpaca/$temperature/$reduction/$learning_rate/$NAME-$smoothing \
                        --dataset_mix_dir=/work/jerry/output-temperature-alpaca/$NAME \
                        --dataset_mixer='{"arazd/tulu_stanford_alpaca": 1.0}' \
                        --reduce_loss=$reduction \
                        --checkpointing_steps=250
                        #--report_to=tensorboard,wandb
                    else
                        echo "File '/work/jerry/output-temperature-alpaca/$temperature/$reduction/$learning_rate/$NAME-$smoothing/config.json' already exists."
                    fi
                done

                BATCH_SIZE_PER_GPU=4
                GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
                for config in \
                    configs/train_configs/tulu3/gemma_sft_2b.yaml \
                    configs/train_configs/tulu3/tulu3_2_sft_3b.yaml \
                    configs/train_configs/tulu3/qwen2_5_sft_1_5b.yaml \
                    configs/train_configs/tulu3/qwen2_5_sft_3b.yaml \
                ;
                do
                    NAME=$(echo $config | cut -d'.' -f 1 | rev | cut -d'/' -f 1 | rev)
                    if [ ! -f /work/jerry/output-temperature-alpaca/$temperature/$reduction/$learning_rate/$NAME-$smoothing/config.json ]; then
                        accelerate launch \
                        --mixed_precision bf16 \
                        --main_process_port 29600 \
                        --num_machines 1 \
                        --num_processes $NUM_GPUS \
                        --use_deepspeed \
                        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
                        open_instruct/finetune-working.py \
                        $config \
                        --temperature=$temperature \
                        --per_device_train_batch_size=$BATCH_SIZE_PER_GPU \
                        --gradient_accumulation_steps=$GRADIENT_ACC_STEPS \
                        --label_smoothing=$smoothing \
                        --learning_rate=$learning_rate \
                        --run_name=$NAME-$smoothing \
                        --output_dir=/work/jerry/output-temperature-alpaca/$temperature/$reduction/$learning_rate/$NAME-$smoothing \
                        --dataset_mix_dir=/work/jerry/output-temperature-alpaca/$NAME \
                        --dataset_mixer='{"arazd/tulu_stanford_alpaca": 1.0}' \
                        --reduce_loss=$reduction \
                        --checkpointing_steps=250
                        #--report_to=tensorboard,wandb
                    else
                        echo "File '/work/jerry/output-temperature-alpaca/$temperature/$reduction/$learning_rate/$NAME-$smoothing/config.json' already exists."
                    fi
                done
            done
        done
    done
done