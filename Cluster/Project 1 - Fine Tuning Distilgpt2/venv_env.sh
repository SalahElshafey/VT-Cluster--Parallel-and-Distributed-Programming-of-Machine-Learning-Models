#!/usr/bin/env bash
# venv_env.sh — safe activator for DistilGPT2 LoRA DDP jobs

# Be strict, but keep nounset OFF until after activation
set -Ee -o pipefail

echo "=== [venv_env.sh] Starting environment setup on $(hostname) ==="

# --- Clean/prime Python env so activate scripts won't break ---
unset PYTHONHOME || true
: "${PYTHONPATH:=}"        # define (empty) if unset to avoid 'unbound variable'
export PATH="/usr/bin:/bin:$PATH"   # guard against minimal PATH on remote srun shells

# --- Prefer cluster-wide env, else fall back to your local one ---
if [[ -f "/opt/llamaenv/bin/activate" ]]; then
  echo "[venv_env.sh] Using /opt/llamaenv"
  # shellcheck disable=SC1091
  source /opt/llamaenv/bin/activate
elif [[ -f "$HOME/llamaenv_local/bin/activate" ]]; then
  echo "[venv_env.sh] Using ~/llamaenv_local"
  # shellcheck disable=SC1091
  source "$HOME/llamaenv_local/bin/activate"
else
  echo "[venv_env.sh] ERROR: No activate script at /opt/llamaenv or ~/llamaenv_local" >&2
  exit 90
fi

# Now it’s safe to enable nounset for the rest of the job
set -u

# --- Core knobs (CPU-only DDP over gloo) ---
export CUDA_VISIBLE_DEVICES=""
export TORCH_DISTRIBUTED_BACKEND=gloo
export GLOO_SOCKET_TIMEOUT=600
export TOKENIZERS_PARALLELISM=false

# --- Hugging Face caches (reduce shared-FS storms) ---
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# --- Sanity probe: stdlib & core libs ---
python - <<'PY'
import sys, sysconfig, math
print("[pyenv]", sys.version.split()[0], "stdlib:", sysconfig.get_paths().get("stdlib"))
assert hasattr(math, "sqrt"), "stdlib 'math' missing"
try:
    import torch, transformers, datasets, peft
    print("[mods] torch", torch.__version__,
          "transformers", transformers.__version__,
          "datasets", datasets.__version__,
          "peft", peft.__version__)
except Exception as e:
    raise SystemExit("[mods] import failed: %r" % (e,))
PY

echo "=== [venv_env.sh] Environment ready on $(hostname) ==="
