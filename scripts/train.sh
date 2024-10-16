#!/bin/bash

#Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Define training tasks
train_tasks=(
    "accelerate launch --config_file ds_config_4gpu.yaml train.py --cfg-path /workspace/project/Research_3D_Aff/Programme_affllm_code_chs/configs/phi_fianl/phi_train_main_4gpu_3e-4.yaml > /workspace/project/Research_3D_Aff/Programme_affllm_code_chs/final/log_0910/1.log 2>&1"
    "accelerate launch --config_file ds_config_4gpu.yaml train.py --cfg-path /workspace/project/Research_3D_Aff/Programme_affllm_code_chs/configs/phi_fianl/phi_train_main_4gpu_3e-4.yaml > /workspace/project/Research_3D_Aff/Programme_affllm_code_chs/final/log_0910/2.log 2>&1"
)

# Execute each training task in sequence
for task in "${train_tasks[@]}"; do
    echo "Execute task: $task"
    eval $task
    if [ $? -eq 0 ]; then
        echo "Task completed successfully: $task"
    else
        echo "Task failed: $task"
        exit 1
    fi
done

echo "All tasks completed."