#!/bin/bash
source /opt/llamaenv/bin/activate
export PYTHONPATH=/opt/llamaenv/lib/python3.11/site-packages
export DS_BUILD_OPS=0
export TORCH_DISTRIBUTED_BACKEND=gloo
export ACCELERATE_DISABLE=false
export CUDA_VISIBLE_DEVICES=""
export USE_CPU_ONLY=true
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets_$HOSTNAME

