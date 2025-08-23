#!/bin/bash

source /home/yhanafy/Summer-Training-2025/Dgpt2/distilgpt2-finetune/venv_env.sh
export WORLD_SIZE=20                       # 20 machines × 1 processes
time accelerate launch \
  --num_processes 1 \
  --main_process_ip hpc11 \
  --main_process_port 12345 \
  --machine_rank ${RANK} \
  --num_machines 20 \
  train_distilgpt2_lora_deepspeed.py > accelerate_${RANK}.log  2>&1 &

