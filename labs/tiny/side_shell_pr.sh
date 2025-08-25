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
VIS_REPORT=$(srun --export=ALL -N "$NNODES" -n "$NNODES" bash -lc '
  echo -n "$HOSTNAME "
  [[ -d "'"$PROJECT_DIR"'" ]] && printf "proj=ok " || printf "proj=missing "
  [[ -d "'"$OFFLINE_ROOT"'/pkgs" ]] && printf "pkgs=ok\n" || printf "pkgs=missing\n"
')
echo "$VIS_REPORT"

if echo "$VIS_REPORT" | grep -q "missing"; then
  echo "[stage] staging project & pkgs via sbcast (to per-node /tmp)"

  TMP_TARS=$(mktemp -d)
  tar -C "$(dirname "$PROJECT_DIR")" -cf "$TMP_TARS/project.tar" "$(basename "$PROJECT_DIR")"
  tar -C "$OFFLINE_ROOT"            -cf "$TMP_TARS/pkgs.tar"     "pkgs"

  sbcast --compress --force "$TMP_TARS/project.tar" /tmp/project_$SLURM_JOB_ID.tar
  sbcast --compress --force "$TMP_TARS/pkgs.tar"    /tmp/pkgs_$SLURM_JOB_ID.tar

  srun --export=ALL -N "$NNODES" -n "$NNODES" bash -lc '
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

  # make these visible to Python
  export RUN_OFFLINE_ROOT RUN_PROJECT_DIR RUN_TMP

  # crucial: let Python find offline site-packages *before* importing torch.*
  export PYTHONPATH="$RUN_OFFLINE_ROOT/pkgs:$RUN_PROJECT_DIR:${PYTHONPATH:-}"

  # caches & threading
  export HF_HOME="$HOME/.cache/hf"
  # TRANSFORMERS_CACHE is deprecated; HF_HOME is enough, so we won’t set it.
  export OMP_NUM_THREADS=2
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_DIST_BACKEND=gloo
'

# ============ 1) PER-NODE SANITY ============
srun --export=ALL --nodes="$NNODES" --ntasks="$NNODES" bash -lc "
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
import torch, transformers
print('NODE', socket.gethostname(), 'OK -> PY', sys.version.split()[0],
      'torch', torch.__version__, 'tfm', transformers.__version__, 'root', root)
PY
  \"\$PYBIN\" /tmp/_sanity.py && rm -f /tmp/_sanity.py
"

# ============ 2) DISTRIBUTED LAUNCH ============
srun --export=ALL --nodes="$NNODES" --ntasks="$NNODES" --kill-on-bad-exit=1 bash -lc "
  set -e
  $RUN_WRAP_ENV
  PYBIN=\$(command -v python3.11 || command -v python3 || command -v python)

  cat >/tmp/_run_wrapper.py << 'PYW'
import os, sys, site, runpy
root = os.environ['RUN_OFFLINE_ROOT']
proj = os.environ['RUN_PROJECT_DIR']
site.addsitedir(os.path.join(root, 'pkgs'))
sys.path.insert(0, proj)
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