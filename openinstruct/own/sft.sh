#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}/..
WORKING_DIR=$( pwd )
pushd ${WORKING_DIR}

# ===== Run =====

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

NUM_NODES=1
GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
MASTER_ADDR=localhost
MASTER_PORT=29600
LOG_RANK=0

echo "===== System Variables ====="
{
    echo "NUM_NODES=$NUM_NODES"
    echo "GPUS_PER_NODE=$GPUS_PER_NODE"
    echo "WORLD_SIZE=$WORLD_SIZE"
    echo "MASTER_ADDR=$MASTER_ADDR"
    echo "MASTER_PORT=$MASTER_PORT"
    echo "LOG_RANK=$LOG_RANK"
} | column -t -s=
echo "============================"

# ===== Default options =====
MODEL=llm
DATASET=HuggingFaceFW/fineweb-edu
DATASET_SPLIT=sample-10BT
DATASET_NAME=$(echo ${DATASET} | cut -d'/' -f 2-)
TOTAL_BATCH_SIZE=128
BATCH_SIZE_PER_GPU=2
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

CONFIG="configs/train_configs/tulu3/tulu3_2_sft_1b.yaml"

echo "===== Training Variables ====="
{
    echo "LENGTH=$LENGTH"
    echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
    echo "BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU"
    echo "GRADIENT_ACC_STEPS=$GRADIENT_ACC_STEPS"
} | column -t -s=
echo "============================"

wandb disabled

# ===== Launch =====
for SMOOTHING in 0.0 0.1;
do

    if [ ! -f outputs/${MODEL}/ft/${DATASET_NAME}-${DATASET_SPLIT}-oh-s$SMOOTHING/config.json ]; then
        BATCH_SIZE_PER_GPU=4
        GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

        accelerate launch \
            --mixed_precision bf16 \
            --num_machines "${NUM_NODES}" \
            --num_processes "${WORLD_SIZE}" \
            --main_process_port "${MASTER_PORT}" \
            --use_deepspeed \
            --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
            "open_instruct/finetune-working.py" \
            ${CONFIG} \
            --timeout=10000000 \
            --model_name_or_path="outputs/${MODEL}/pt/${DATASET_NAME}-${DATASET_SPLIT}" \
            --num_train_epochs=2 \
            --learning_rate=5e-6 \
            --lr_scheduler_type=cosine \
            --weight_decay=0.0 \
            --warmup_ratio=0.03 \
            --per_device_train_batch_size=${BATCH_SIZE_PER_GPU} \
            --gradient_accumulation_steps=${GRADIENT_ACC_STEPS} \
            --reduce_loss="sum" \
            --label_smoothing=${SMOOTHING} \
            --output_dir="outputs/${MODEL}/ft/${DATASET_NAME}-${DATASET_SPLIT}-oh-s$SMOOTHING" \
            --dataset_mix_dir="outputs/${MODEL}/ft-datamix/${DATASET_NAME}-${DATASET_SPLIT}-oh-s$SMOOTHING" \
            --dataset_mixer='{"teknium/OpenHermes-2.5": 1.0}'
    fi

    if [ ! -f outputs/${MODEL}/ft/${DATASET_NAME}-${DATASET_SPLIT}-tulu-s$SMOOTHING/config.json ]; then
        BATCH_SIZE_PER_GPU=2
        GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

        accelerate launch \
            --mixed_precision bf16 \
            --num_machines "${NUM_NODES}" \
            --num_processes "${WORLD_SIZE}" \
            --main_process_port "${MASTER_PORT}" \
            --use_deepspeed \
            --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
            "open_instruct/finetune-working.py" \
            ${CONFIG} \
            --timeout=10000000 \
            --model_name_or_path="outputs/${MODEL}/pt/${DATASET_NAME}-${DATASET_SPLIT}" \
            --num_train_epochs=2 \
            --learning_rate=5e-6 \
            --lr_scheduler_type=cosine \
            --weight_decay=0.0 \
            --warmup_ratio=0.03 \
            --per_device_train_batch_size=${BATCH_SIZE_PER_GPU} \
            --gradient_accumulation_steps=${GRADIENT_ACC_STEPS} \
            --reduce_loss="sum" \
            --label_smoothing=${SMOOTHING} \
            --output_dir="outputs/${MODEL}/ft/${DATASET_NAME}-${DATASET_SPLIT}-tulu-s$SMOOTHING" \
            --dataset_mix_dir="outputs/${MODEL}/ft-datamix/${DATASET_NAME}-${DATASET_SPLIT}-tulu-s$SMOOTHING" \
            --dataset_mixer='{"allenai/tulu-3-sft-mixture": 1.0}'
    fi

    if [ ! -f outputs/${MODEL}/ft/${DATASET_NAME}-${DATASET_SPLIT}-ap-s$SMOOTHING/config.json ]; then
        BATCH_SIZE_PER_GPU=8
        GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
        
        accelerate launch \
            --mixed_precision bf16 \
            --num_machines "${NUM_NODES}" \
            --num_processes "${WORLD_SIZE}" \
            --main_process_port "${MASTER_PORT}" \
            --use_deepspeed \
            --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
            "open_instruct/finetune-working.py" \
            ${CONFIG} \
            --timeout=10000000 \
            --model_name_or_path="outputs/${MODEL}/pt/${DATASET_NAME}-${DATASET_SPLIT}" \
            --num_train_epochs=2 \
            --learning_rate=5e-6 \
            --lr_scheduler_type=cosine \
            --weight_decay=0.0 \
            --warmup_ratio=0.03 \
            --per_device_train_batch_size=${BATCH_SIZE_PER_GPU} \
            --gradient_accumulation_steps=${GRADIENT_ACC_STEPS} \
            --reduce_loss="sum" \
            --label_smoothing=${SMOOTHING} \
            --output_dir="outputs/${MODEL}/ft/${DATASET_NAME}-${DATASET_SPLIT}-ap-s$SMOOTHING" \
            --dataset_mix_dir="outputs/${MODEL}/ft-datamix/${DATASET_NAME}-${DATASET_SPLIT}-ap-s$SMOOTHING" \
            --dataset_mixer='{"arazd/tulu_stanford_alpaca": 1.0}'
    fi
done

popd