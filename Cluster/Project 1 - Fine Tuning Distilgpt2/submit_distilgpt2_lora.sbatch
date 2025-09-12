#!/bin/bash
#SBATCH -J gpt2lora
#SBATCH -p torch
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -Eeuo pipefail
trap 'echo "[FATAL] line $LINENO rc=$?" >&2' ERR
umask 027
export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin:/sbin}"

# -------- Inputs --------
DATASET="${1:-tiny}"    # tiny | medium
WORKDIR="${WORKDIR:-$HOME/project/Cluster/Project 1 - Fine Tuning Distilgpt2}"
VENV_ACT="${VENV_ACT:-$WORKDIR/venv_env.sh}"
SEQ_LEN="${SEQ_LEN:-256}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-32}"
LR="${LR:-5e-5}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

export WORKDIR VENV_ACT SEQ_LEN EPOCHS BATCH ACCUM LR OMP_NUM_THREADS

# -------- Paths --------
[[ -d "$WORKDIR" ]] || { echo "ERROR: WORKDIR not found: $WORKDIR" >&2; exit 10; }
[[ -f "$WORKDIR/finetune_lora_distilgpt2.py" ]] || { echo "ERROR: training script missing in WORKDIR"; ls -l "$WORKDIR"; exit 11; }
mkdir -p "$WORKDIR/logs"

# -------- Dataset --------
case "$DATASET" in
  tiny)   DATA_FILE="/data/tiny_openwebtext.txt" ;;
  medium) DATA_FILE="/data/medium_openwebtext.txt" ;;
  *) echo "ERROR: DATASET must be tiny|medium (got '$DATASET')" >&2; exit 12 ;;
esac
export DATA_FILE

# -------- DDP (SLURM-native env://) + networking --------
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${SLURM_NNODES}"
IFACE_DEFAULT="$(ip route | awk '/default/ {print $5; exit}')"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens160}"
[[ -n "$GLOO_SOCKET_IFNAME" ]] || export GLOO_SOCKET_IFNAME="$IFACE_DEFAULT"

export MASTER_ADDR MASTER_PORT NNODES
export TORCH_DISTRIBUTED_BACKEND="${TORCH_DISTRIBUTED_BACKEND:-gloo}"
export GLOO_SOCKET_TIMEOUT="${GLOO_SOCKET_TIMEOUT:-1800}"
export TORCH_DIST_INIT_BARRIER=1

# -------- HF caches --------
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# -------- Logs & outputs --------
LOGDIR="$WORKDIR/logs/${SLURM_JOB_ID}"
OUT_ROOT="${OUT_ROOT:-$HOME/finetuned}"
export OUT_ROOT LOGDIR
mkdir -p "$LOGDIR" "$OUT_ROOT"
ln -sfn "$LOGDIR" "$WORKDIR/logs/latest"

echo "[SLURM] job=$SLURM_JOB_ID nodes=$NNODES dataset=$DATASET master=$MASTER_ADDR:$MASTER_PORT iface=$GLOO_SOCKET_IFNAME"
scontrol show hostnames "$SLURM_JOB_NODELIST"

cat > "$LOGDIR/meta.json" <<JSON
{
  "job_id": "$SLURM_JOB_ID",
  "nnodes": $NNODES,
  "dataset": "$DATASET",
  "data_file": "$DATA_FILE",
  "seq_len": $SEQ_LEN,
  "epochs": $EPOCHS,
  "batch": $BATCH,
  "accum": $ACCUM,
  "lr": $LR,
  "master_addr": "$MASTER_ADDR",
  "master_port": $MASTER_PORT,
  "gloo_iface": "${GLOO_SOCKET_IFNAME}"
}
JSON

JOB_T0=$(date +%s)

# -------- Preflight (all nodes) --------
srun --kill-on-bad-exit=1 --ntasks=$NNODES --ntasks-per-node=1 \
  /bin/bash --noprofile --norc -c '
    set -euo pipefail
    echo "[Preflight] start on $(hostname)"
    : "${VENV_ACT:?VENV_ACT not set}"
    : "${DATA_FILE:?DATA_FILE not set}"
    : "${OUT_ROOT:?OUT_ROOT not set}"
    test -r "${VENV_ACT:-}"  || { echo "[Preflight] Missing venv activator: ${VENV_ACT:-<unset>}" >&2; exit 21; }
    test -r "${DATA_FILE:-}" || { echo "[Preflight] Cannot read dataset: ${DATA_FILE:-<unset>}" >&2; exit 22; }
    test -w "${OUT_ROOT:-}"  || { echo "[Preflight] OUT_ROOT not writable: ${OUT_ROOT:-<unset>}" >&2; exit 23; }
    echo "[Preflight] done on $(hostname)"
  '

# -------- Launch: one Python rank per node (no torchrun) --------
set +e
srun --kill-on-bad-exit=1 \
     --ntasks=$NNODES \
     --ntasks-per-node=1 \
     --cpu-bind=cores \
     --chdir="$WORKDIR" \
  /bin/bash --noprofile --norc -c '
    set -euo pipefail
    cd "$WORKDIR"
    echo "[Launch] PWD=$(pwd) node=$(hostname) venv: $VENV_ACT OMP_NUM_THREADS=$OMP_NUM_THREADS"

    unset PYTHONHOME PYTHONPATH
    set +u; source "$VENV_ACT"; set -u
    hash -r
    PY=$(command -v python)
    echo "[Launch] python=$PY"

    "$PY" - <<'"'"'PY'"'"' || { echo "[Launch] python import failed" >&2; exit 34; }
import sys, sysconfig, math
print("[pyenv]", sys.version.split()[0], "math?", hasattr(math, "sqrt"))
try:
    import torch, transformers, datasets, peft
    print("[mods] torch", torch.__version__, "transformers", transformers.__version__, "datasets", datasets.__version__, "peft", peft.__version__)
except Exception as e:
    print("[mods] import fail:", e); raise
PY

    [[ -f "./finetune_lora_distilgpt2.py" ]] || { echo "[Launch] script missing in PWD"; ls -la; exit 33; }

    # ----- Map SLURM -> env:// ranks -----
    export MASTER_ADDR="$MASTER_ADDR"
    export MASTER_PORT="$MASTER_PORT"
    export WORLD_SIZE="${SLURM_NTASKS:-$NNODES}"
    export RANK="${SLURM_PROCID}"
    export LOCAL_RANK="${SLURM_LOCALID}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
    export GLOO_SOCKET_TIMEOUT="${GLOO_SOCKET_TIMEOUT}"
    export TORCH_DIST_INIT_BARRIER="${TORCH_DIST_INIT_BARRIER}"
    echo "[Launch] RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE MASTER=${MASTER_ADDR}:${MASTER_PORT} IFACE=$GLOO_SOCKET_IFNAME"

    "$PY" ./finetune_lora_distilgpt2.py \
      --dataset "$DATASET" \
      --data_file "$DATA_FILE" \
      --seq_len "$SEQ_LEN" \
      --epochs "$EPOCHS" \
      --batch "$BATCH" \
      --accum "$ACCUM" \
      --lr "$LR" \
      --logdir "$LOGDIR" \
      --out_root "$OUT_ROOT"
  '
SRUN_RC=$?
set -e

# -------- Wrap up --------
JOB_T1=$(date +%s)
WALL=$((JOB_T1 - JOB_T0))
echo "$WALL" > "$LOGDIR/wallclock_seconds.txt"

if command -v jq >/dev/null 2>&1; then
  jq --arg status "$([ "$SRUN_RC" -eq 0 ] && echo success || echo failure)" \
     --argjson wall $WALL \
     '. + {status: $status, wall_seconds: $wall}' \
     "$LOGDIR/meta.json" > "$LOGDIR/meta.final.json" || true
fi

echo "[SLURM] completed: logs=$LOGDIR wall=${WALL}s status=$([ "$SRUN_RC" -eq 0 ] && echo OK || echo FAIL) (rc=$SRUN_RC)"
exit "$SRUN_RC"
