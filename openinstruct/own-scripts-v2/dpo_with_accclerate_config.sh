#!/bin/bash

# example usage
# sh scripts/dpo_train_with_accelerate_config.sh 8 configs/train_configs/dpo/default.yaml

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
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

for config in \
    configs/train_configs/own_dpo/llama3_dpo_8b.yaml \
    configs/train_configs/own_dpo/tulu3_dpo_8b.yaml \
;
do
    NAME=$(echo $config | cut -d'.' -f 1 | rev | cut -d'/' -f 1 | rev)

    # You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
    # but it will trade off speed.
    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        open_instruct/dpo_tune.py \
        $config \
        --output_dir=/work/jerry/output_dpo/$name \
        --per_device_train_batch_size=$BATCH_SIZE_PER_GPU \
        --gradient_accumulation_steps=$GRADIENT_ACC_STEPS \
done