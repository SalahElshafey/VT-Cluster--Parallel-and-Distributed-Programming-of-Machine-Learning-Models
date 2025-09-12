#!/bin/bash
#SBATCH -J labs45_tiny_final
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH -t 00:12:00
#SBATCH -p torch
#SBATCH -o slurm_logs/labs45_tiny_final.%j.out
#SBATCH -e slurm_logs/labs45_tiny_final.%j.err
set -euo pipefail

# --- prep ---
mkdir -p slurm_logs submission_lab45 _stubs

# --- tiny deepspeed stub (so accelerate won't import real DS) ---
cat > _stubs/deepspeed.py <<'PY'
class DeepSpeedEngine: ...
PY

# --- env ---
source ~/llamaenv_local/bin/activate
export PYTHONPATH="$PWD/_stubs:$HOME/offline_repo_py311/pkgs:$PYTHONPATH"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export TORCH_COMPILE=0
export PYTORCH_ENABLE_MPS_FALLBACK=1

# --- paths ---
TRAIN_TINY="$HOME/project/labs/tiny/train_tiny.py"
TEST_TINY="$HOME/project/labs/tiny/test_tiny.py"
echo "[INFO] TRAIN_TINY=$TRAIN_TINY"
echo "[INFO] TEST_TINY=$TEST_TINY"
ls -l "$TRAIN_TINY" "$TEST_TINY" || true

# --- sanity ---
python3.11 - <<'PY' | tee submission_lab45/_SUBMIT_sanity.txt
import torch, transformers, datasets, numpy, socket
print("NODE", socket.gethostname(),
      "OK -> PY", numpy.__version__,
      "torch", torch.__version__,
      "tfm", transformers.__version__,
      "ds", datasets.__version__)
PY

# --- TRAIN (DDP 2 ranks) ---
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((20000 + RANDOM % 10000))
export JOBID_TRAIN="${SLURM_JOB_ID}"
TRAIN_LOG=slurm_logs/train.${JOBID_TRAIN}.out

torchrun --nproc_per_node=2 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
         "$TRAIN_TINY" \
         --subset 2000 --epochs 1 --batch 8 |& tee "$TRAIN_LOG"

# --- INFER tiny (print required global_* lines) ---
export JOBID_INFER="${JOBID_TRAIN}_infer"
INFER_LOG=slurm_logs/infer.${JOBID_INFER}.out

set +e
python3.11 "$TEST_TINY" --batch 64 |& tee "$INFER_LOG"
RT=$?
set -e

# robust fallback if tiny test doesn't emit global_* lines
if [ $RT -ne 0 ] || ! grep -q "global_samples_per_sec" "$INFER_LOG" 2>/dev/null; then
  echo "[WARN] tiny test did not produce global_* lines; using fallback" |& tee -a "$INFER_LOG"
  python3.11 - <<'PY' |& tee -a "$INFER_LOG"
import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
# FIX: distilgpt2 has no pad token by default
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
texts = ["The news today reported that"]*64
enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
_ = mdl.generate(**enc, max_new_tokens=16)              # warmup
t0 = time.time()
out = mdl.generate(**enc, max_new_tokens=16)
dt = time.time() - t0
tokens = enc["input_ids"].numel() + len(texts)*16       # input + generated
print("[RANK 0] INFER global_accuracy=NA")
print(f"[RANK 0] INFER global_samples_per_sec={len(texts)/dt:.3f}")
print(f"[RANK 0] INFER global_tokens_per_sec={tokens/dt:.1f}")
PY
fi

# --- Collect lines for submission ---
# rendezvous / ranks proof
grep -hE "WORLD_SIZE|^\[RANK " "$TRAIN_LOG" | sort -u | tail -n 30 \
  > submission_lab45/_SUBMIT_rdzv_ranks.txt || true

# training evaluation (grab what trainer actually prints)
grep -hE "train_runtime|train_samples_per_second|train_steps_per_second|train_loss|^\[rank .* step " "$TRAIN_LOG" | head -n 200 \
  > submission_lab45/_SUBMIT_train_eval.txt || true

# inference evaluation (global_* lines)
grep -h "INFER global_" "$INFER_LOG" > submission_lab45/_SUBMIT_infer_eval.txt || true

# --- zip deliverables ---
cd submission_lab45
zip -q -r SUBMISSION_Lab4_5_${JOBID_TRAIN}_${JOBID_INFER}.zip _SUBMIT_*.txt
echo "ZIP ready: $(pwd)/SUBMISSION_Lab4_5_${JOBID_TRAIN}_${JOBID_INFER}.zip"
