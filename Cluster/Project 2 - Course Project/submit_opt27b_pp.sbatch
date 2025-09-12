#!/bin/bash
#SBATCH --job-name=opt27b_pp_lora
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7700M
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# 1) venv created by venv_env.sh
export VENV_ACT="$HOME/llamaenv_local/bin/activate"

# 2) model + DeepSpeed config (uses activation checkpointing on CPU)
export MODEL_NAME="facebook/opt-2.7b"
export DS_CFG="$PWD/deepspeed_pp_zero1_cpu_activ.json"

# 3) dataset: pre-tokenized HF dataset dir (fast) OR raw text file
#    >>>>>>>>>>>>  CHANGE THIS PATH to your dataset <<<<<<<<<<<<
export DATA_FILE="/path/to/openwebtext_tok_dir"

# 4) training knobs (small micro-batch, big accumulation)
export SEQ_LEN=512
export EPOCHS=1
export BATCH=1
export ACCUM=96
export LR=5e-5
export LOGDIR="logs/${SLURM_JOB_ID}"
export OUT_ROOT="$HOME/finetuned"

# 5) torch.distributed rendezvous for env://
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

# 6) number of pipeline stages == world size
export PIPELINE_PARALLEL_SIZE=$SLURM_NTASKS

# 7) HF datasets hygiene + threads
export HF_DATASETS_IN_MEMORY=0
export HF_DATASETS_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LOGDIR"

# Optional wallclock timing
date +%s > "${LOGDIR}/t_start"
srun -l ./run_node.sh
date +%s > "${LOGDIR}/t_end"

python - <<'PY'
import os, json
ld=os.environ.get("LOGDIR",".")
try:
    t0=int(open(os.path.join(ld,"t_start")).read().strip())
    t1=int(open(os.path.join(ld,"t_end")).read().strip())
    open(os.path.join(ld,"wallclock_seconds.txt"),"w").write(str(t1-t0))
except: pass
PY
