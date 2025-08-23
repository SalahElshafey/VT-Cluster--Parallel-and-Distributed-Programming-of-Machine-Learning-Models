#!/usr/bin/env bash
set -euo pipefail

# ===================== SETTINGS =====================
PARTITION=torch
JOB_NODES=10
NPROC_PER_NODE=2
TIME=00:30:00

export PROJECT_DIR="$HOME/mrmito/project"
export OFFLINE_ROOT="$HOME/offline_repo_py311"   # must contain: $OFFLINE_ROOT/pkgs

# Sanity on shared pkgs path
if [[ ! -d "$OFFLINE_ROOT/pkgs" ]]; then
  echo "ERROR: $OFFLINE_ROOT/pkgs not found. Build your offline pkgs there first."; exit 1
fi
if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: $PROJECT_DIR not found."; exit 1
fi

# ===================== SLURM ALLOCATION =====================
salloc --partition="$PARTITION" --nodes="$JOB_NODES" --ntasks-per-node=1 --cpus-per-task=4 --time="$TIME"

# Rendezvous for this allocation
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export NNODES=${SLURM_NNODES:-$(scontrol show hostnames "$SLURM_NODELIST" | wc -l)}
echo "NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# ===================== 1) PER-NODE SANITY =====================
srun --export=ALL -N "$NNODES" -n "$NNODES" --chdir "$PROJECT_DIR" bash -lc '
  set -e
  PYBIN=$(command -v python3.11 || command -v python3 || command -v python)
  cat >/tmp/_sanity.py << "PY"
import os, sys, socket, site
site.addsitedir(os.path.join(os.environ["OFFLINE_ROOT"], "pkgs"))   # ensure offline pkgs are on sys.path
sys.path.insert(0, os.environ["PROJECT_DIR"])                       # add your repo
import torch, transformers
print("NODE", socket.gethostname(), "OK -> PY", sys.version.split()[0],
      "torch", torch.__version__, "tfm", transformers.__version__)
PY
  "$PYBIN" /tmp/_sanity.py && rm -f /tmp/_sanity.py
'

# ===================== 2) DISTRIBUTED LAUNCH =====================
srun --export=ALL --nodes="$NNODES" --ntasks="$NNODES" --chdir "$PROJECT_DIR" --kill-on-bad-exit=1 bash -lc '
  set -e
  PYBIN=$(command -v python3.11 || command -v python3 || command -v python)

  # Writable caches
  export HF_HOME="$HOME/.cache/hf"
  export TRANSFORMERS_CACHE="$HF_HOME"

  # tiny wrapper that injects paths INSIDE Python and then runs your training script
  cat >/tmp/_run_wrapper.py << "PYW"
import os, sys, site, runpy
site.addsitedir(os.path.join(os.environ["OFFLINE_ROOT"], "pkgs"))    # offline pkgs
sys.path.insert(0, os.path.abspath(os.environ["PROJECT_DIR"]))       # your repo
# execute labs/tiny/train_tiny.py with passed args
train_path = os.path.join(os.environ["PROJECT_DIR"], "labs/tiny/train_tiny.py")
sys.argv = [train_path] + sys.argv[1:]
runpy.run_path(train_path, run_name="__main__")
PYW

  "$PYBIN" -m torch.distributed.run \
    --nnodes='"$NNODES"' \
    --nproc_per_node='"$NPROC_PER_NODE"' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    --rdzv_id="$SLURM_JOB_ID" \
    --node_rank="$SLURM_NODEID" \
    /tmp/_run_wrapper.py --subset 2000 --epochs 3
'

# set defaults if missing
: "${NPROC_PER_NODE:=2}"
: "${NNODES:=$(scontrol show hostnames "$SLURM_NODELIST" | wc -l)}"
: "${MASTER_ADDR:=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)}"
: "${MASTER_PORT:=29500}"

# (optional) confirm
echo "NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE MASTER=$MASTER_ADDR:$MASTER_PORT"

srun --export=ALL --nodes="$NNODES" --ntasks="$NNODES" --chdir "$PROJECT_DIR" --kill-on-bad-exit=1 bash -lc '
  set -e
  PYBIN=$(command -v python3.11 || command -v python3 || command -v python)

  export PYTHONPATH="'"$OFFLINE_ROOT"'/pkgs:'"$PROJECT_DIR"':$PYTHONPATH"
  export HF_HOME="$HOME/.cache/hf"
  export TRANSFORMERS_CACHE="$HF_HOME"
  export OMP_NUM_THREADS=2
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_DIST_BACKEND=gloo

  echo "torchrun: nnodes='"$NNODES"' nproc_per_node='"$NPROC_PER_NODE"' node_rank=$SLURM_NODEID rdzv='"$MASTER_ADDR:$MASTER_PORT"'"

  "$PYBIN" -m torch.distributed.run \
    --nnodes='"$NNODES"' \
    --nproc_per_node='"$NPROC_PER_NODE"' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    --rdzv_id="$SLURM_JOB_ID" \
    --node_rank="$SLURM_NODEID" \
    labs/tiny/train_tiny.py --subset 2000 --epochs 3
'
