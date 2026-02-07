#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}/..
WORKING_DIR=$( pwd )
pushd ${WORKING_DIR}

# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH             
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True   # For cmmlu dataset

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Enable async error handling for multi-GPU communication to avoid deadlocks
export NCCL_DEBUG=warn                      # Show NCCL warnings for better diagnosis without flooding logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Provide detailed logging for PyTorch distributed debugging

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

# ===== Environment =====
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH=.

# ===== Default options =====
ACCELERATE_CONFIG="fsdp"

MODEL=llm
DATASET=HuggingFaceFW/fineweb-edu
DATASET_SPLIT=sample-100BT
DATASET_NAME=$(echo ${DATASET} | cut -d'/' -f 2-)
LENGTH=2048
TOTAL_BATCH_SIZE=$(( 524288 / $LENGTH ))
BATCH_SIZE_PER_GPU=4
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

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
# torchrun --nproc_per_node=${GPUS_PER_NODE} \
#     --nnodes=${NUM_NODES} \
#     --rdzv_backend c10d \
#     --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
#     --local-ranks-filter ${LOG_RANK} \
#     --role rank \
#     --tee 3 \

# Salesforce/wikitext
# wikitext-103-raw-v1

rm -rf "outputs/${MODEL}/pt/${DATASET_NAME}-${DATASET_SPLIT}"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines "${NUM_NODES}" \
    --num_processes "${WORLD_SIZE}" \
    --main_process_port "${MASTER_PORT}" \
    "open_instruct/pt.py" \
    --ddp_backend nccl \
    --ddp_timeout 100000000 \
    --config_name "models/${MODEL}" \
    --tokenizer_name "models/${MODEL}" \
    --dataset_name ${DATASET} \
    --dataset_config_name ${DATASET_SPLIT} \
    --do_train \
    --block_size ${LENGTH} \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --warmup_ratio 0.01 \
    --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --per_device_eval_batch_size ${BATCH_SIZE_PER_GPU} \
    --gradient_accumulation_steps ${GRADIENT_ACC_STEPS} \
    --output_dir "outputs/${MODEL}/pt/${DATASET_NAME}-${DATASET_SPLIT}" \
    --save_steps 0.1
