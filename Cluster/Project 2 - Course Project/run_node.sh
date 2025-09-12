#!/usr/bin/env bash
set -Eeuo pipefail
echo "[dbg] node=$(hostname) START"
echo "[dbg] VENV_ACT=$VENV_ACT"
echo "[dbg] DATA_FILE=$DATA_FILE"
echo "[dbg] DS_CFG=$DS_CFG"
echo "[dbg] MODEL_NAME=$MODEL_NAME"
echo "[dbg] MASTER=$MASTER_ADDR:$MASTER_PORT"

# stage dataset locally to tmp so every node can read fast
STAGE_DIR="/tmp/openwebtext_tok_${SLURM_JOB_ID:-$$}"
if [[ -d "$DATA_FILE" ]]; then
  mkdir -p "$STAGE_DIR"
  rsync -a --delete "$DATA_FILE/" "$STAGE_DIR/" || cp -r "$DATA_FILE"/* "$STAGE_DIR"/
  export DATA_FILE="$STAGE_DIR"
  echo "[stage] Dataset staged to $DATA_FILE"
fi

source "$VENV_ACT"
export TORCH_DISTRIBUTED_BACKEND=gloo
export CUDA_VISIBLE_DEVICES=""

python ./finetune_lora_opt_pp.py \
  --model_name "$MODEL_NAME" \
  --data_file "$DATA_FILE" \
  --seq_len "$SEQ_LEN" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --accum "$ACCUM" \
  --lr "$LR" \
  --ds_cfg "$DS_CFG" \
  --logdir "$LOGDIR" \
  --out_root "$OUT_ROOT"
