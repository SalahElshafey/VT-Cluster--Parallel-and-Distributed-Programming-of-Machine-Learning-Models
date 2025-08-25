cat > ~/launch_tiny.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

# ============ SETTINGS ============
PARTITION=torch
JOB_NODES=10
NPROC_PER_NODE=2
TIME=00:30:00
SALLOC_OPTS="${SALLOC_OPTS:-}"   # e.g., --exclude=hpc42

export PROJECT_DIR="$HOME/mrmito/project"
export OFFLINE_ROOT="$HOME/offline_repo_py311"

[[ -d "$OFFLINE_ROOT/pkgs" ]] || { echo "ERROR: $OFFLINE_ROOT/pkgs missing"; exit 1; }
[[ -d "$PROJECT_DIR"       ]] || { echo "ERROR: $PROJECT_DIR missing"; exit 1; }

mkdir -p "$HOME/slurm_logs"

# ============ BOOTSTRAP UNDER SALLOC ============
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec salloc --partition="$PARTITION" --nodes="$JOB_NODES" \
              --ntasks-per-node=1 --cpus-per-task=4 --time="$TIME" \
              $SALLOC_OPTS \
              bash -lc "bash '$0' --inside"
fi

echo "[alloc] job=$SLURM_JOB_ID nodes=$SLURM_NODELIST"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export NNODES=${SLURM_NNODES:-$(scontrol show hostnames "$SLURM_NODELIST" | wc -l)}
echo "NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# ============ PREFLIGHT ============
echo "[preflight] checking visibility of PROJECT_DIR and OFFLINE_ROOT/pkgs on all nodes..."
# IMPORTANT: no --output here; we want stdout back for VIS_REPORT
VIS_REPORT=$(srun --export=ALL -N "$NNODES" -n "$NNODES" bash -lc '
  echo -n "$HOSTNAME "
  [[ -d "'"$PROJECT_DIR"'" ]] && printf "proj=ok " || printf "proj=missing "
  [[ -d "'"$OFFLINE_ROOT"'/pkgs" ]] && printf "pkgs=ok\n" || printf "pkgs=missing\n"
')
# show on screen AND save to a file
echo "$VIS_REPORT" | tee "$HOME/slurm_logs/preflight.$SLURM_JOB_ID.txt"

OK_CNT=$(echo "$VIS_REPORT" | grep -c 'proj=ok pkgs=ok' || true)
echo "[preflight] $OK_CNT/$NNODES nodes have both paths"

if [[ "$OK_CNT" -lt "$NNODES" ]]; then
  echo "[stage] staging project & pkgs via sbcast (to per-node /tmp)"

  TMP_TARS=$(mktemp -d)
  tar -C "$(dirname "$PROJECT_DIR")" -cf "$TMP_TARS/project.tar" "$(basename "$PROJECT_DIR")"
  tar -C "$OFFLINE_ROOT"            -cf "$TMP_TARS/pkgs.tar"     "pkgs"

  sbcast --compress --force "$TMP_TARS/project.tar" /tmp/project_$SLURM_JOB_ID.tar
  sbcast --compress --force "$TMP_TARS/pkgs.tar"    /tmp/pkgs_$SLURM_JOB_ID.tar

  srun --label --output="$HOME/slurm_logs/%x.%j.%n.%t.out" \
    --export=ALL -N "$NNODES" -n "$NNODES" bash -lc '
    set -e
    STAGE_DIR="${SLURM_TMPDIR:-/tmp/$USER/slurm_$SLURM_JOB_ID}"
    mkdir -p "$STAGE_DIR"
    tar -xf /tmp/project_'"$SLURM_JOB_ID"'.tar -C "$STAGE_DIR"
    tar -xf /tmp/pkgs_'"$SLURM_JOB_ID"'.tar    -C "$STAGE_DIR"
    echo "$HOSTNAME staged -> $STAGE_DIR"
  '
fi

# Per-node runtime roots using safe fallback
RUN_WRAP_ENV='
  RUN_TMP="${SLURM_TMPDIR:-/tmp/$USER/slurm_$SLURM_JOB_ID}"
  RUN_OFFLINE_ROOT="'"$OFFLINE_ROOT"'"
  RUN_PROJECT_DIR="'"$PROJECT_DIR"'"
  [[ -d "$RUN_PROJECT_DIR" ]]       || RUN_PROJECT_DIR="$RUN_TMP/'"$(basename "$PROJECT_DIR")"'"
  [[ -d "$RUN_OFFLINE_ROOT/pkgs" ]] || RUN_OFFLINE_ROOT="$RUN_TMP"

  # visible to Python
  export RUN_OFFLINE_ROOT RUN_PROJECT_DIR RUN_TMP
  export PYTHONPATH="$RUN_OFFLINE_ROOT/pkgs:$RUN_PROJECT_DIR:${PYTHONPATH:-}"

  # offline/caches & threading
  export HF_HOME="$HOME/.cache/hf"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export OMP_NUM_THREADS=2
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_DIST_BACKEND=gloo
  export PYTHONUNBUFFERED=1
'

# ============ 1) PER-NODE SANITY ============
srun --label --output="$HOME/slurm_logs/%x.%j.%n.%t.out" \
  --export=ALL --nodes="$NNODES" --ntasks="$NNODES" bash -lc "
  set -e
  $RUN_WRAP_ENV
  PYBIN=\$(command -v python3.11 || command -v python3 || command -v python)
  echo \"[\$HOSTNAME] RUN_PROJECT_DIR=\$RUN_PROJECT_DIR RUN_OFFLINE_ROOT=\$RUN_OFFLINE_ROOT\"
  cat >/tmp/_sanity.py << 'PY'
import os, sys, socket, site
root = os.environ['RUN_OFFLINE_ROOT']
proj = os.environ['RUN_PROJECT_DIR']
site.addsitedir(os.path.join(root, 'pkgs'))
sys.path.insert(0, proj)
import torch, transformers, numpy, datasets
print('NODE', socket.gethostname(), 'OK -> PY', sys.version.split()[0],
      'torch', torch.__version__, 'tfm', transformers.__version__,
      'numpy', numpy.__version__, 'datasets', datasets.__version__,
      'root', root)
PY
  \"\$PYBIN\" /tmp/_sanity.py && rm -f /tmp/_sanity.py
"

# ============ 2) DISTRIBUTED LAUNCH ============
srun --label --output="$HOME/slurm_logs/%x.%j.%n.%t.out" \
  --export=ALL --nodes="$NNODES" --ntasks="$NNODES" --kill-on-bad-exit=1 bash -lc "
  set -e
  $RUN_WRAP_ENV
  PYBIN=\$(command -v python3.11 || command -v python3 || command -v python)

  # wrapper injects paths AND applies a NumPy>=2 safety patch for datasets if ever needed
  cat >/tmp/_run_wrapper.py << 'PYW'
import os, sys, site, runpy
root = os.environ['RUN_OFFLINE_ROOT']
proj = os.environ['RUN_PROJECT_DIR']
site.addsitedir(os.path.join(root, 'pkgs'))
sys.path.insert(0, proj)

# ---- Optional safety patch: if NumPy >= 2 and datasets old, make it NumPy-2 safe
try:
    import numpy as _np
    major = int(_np.__version__.split('.')[0])
    if major >= 2:
        try:
            import datasets.formatting.formatting as _fmt
            def _patched_arrow_array_to_numpy(self, array):
                return _np.asarray(array)
            _fmt.NumpyArrowExtractor._arrow_array_to_numpy = _patched_arrow_array_to_numpy
            print('[patch] Applied NumPy 2.x compatibility for datasets formatter', flush=True)
        except Exception as e:
            print('[patch] Skipped datasets patch:', e, flush=True)
except Exception:
    pass
# -------------------------------------------------------------------------------

train_path = os.path.join(proj, 'labs/tiny/train_tiny.py')
sys.argv = [train_path] + sys.argv[1:]
runpy.run_path(train_path, run_name='__main__')
PYW

  echo \"torchrun: nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE node_rank=\$SLURM_NODEID rdzv=$MASTER_ADDR:$MASTER_PORT\"
  \"\$PYBIN\" -m torch.distributed.run \
    --nnodes=\"$NNODES\" \
    --nproc_per_node=\"$NPROC_PER_NODE\" \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\"$MASTER_ADDR:$MASTER_PORT\" \
    --rdzv_id=\"\$SLURM_JOB_ID\" \
    --node_rank=\"\$SLURM_NODEID\" \
    /tmp/_run_wrapper.py --subset 2000 --epochs 3
"

echo "NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE MASTER=$MASTER_ADDR:$MASTER_PORT"

BASH
chmod +x ~/launch_tiny.sh

echo "USE: bash ~/launch_tiny.sh or exlude bad nodes for example SALLOC_OPTS="--exclude=hpc42" bash ~/launch_tiny.sh"