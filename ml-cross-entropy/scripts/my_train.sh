#!/bin/bash

set -e

export WANDB_MODE=offline
export WANDB_PROJECT=cce-training

for name in gemma2 llama3 mistral-nemo phi3.5 llama2 llama3.2-1 llama3.2-3 mistral;
do
    for impl in cce torch_compile;
    do
        for ls in 0.0 0.5 1.0;
        do
            torchrun --standalone --nproc-per-node=8 --module training.train \
                --deepspeed training/zero3.json \
                --model_name $name-$impl-$ls \
                --output_dir /work/jerry/cce-training-checkpoints \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 8 \
                --per_device_eval_batch_size 8 \
                --cross_entropy_impl $impl \
                --label_smoothing $ls \
                --eval_strategy "steps" \
                --eval_steps 1000 \
                --learning_rate 2e-5 \
                --dataloader_num_workers 4 \
                --run_name $name-$impl-$ls \
                --report_to wandb
        done
    done
    torchrun --standalone --nproc-per-node=8 --module training.train \
        --deepspeed training/zero3.json \
        --model_name $name-cce_orig-0.0 \
        --output_dir /work/jerry/cce-training-checkpoints \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --per_device_eval_batch_size 8 \
        --cross_entropy_impl cce \
        --eval_strategy "steps" \
        --eval_steps 1000 \
        --learning_rate 2e-5 \
        --dataloader_num_workers 4 \
        --run_name $name-cce_orig-0.0 \
        --report_to wandb
done