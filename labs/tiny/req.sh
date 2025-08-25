export OFFLINE_ROOT="$HOME/offline_repo_py311"
mkdir -p "$OFFLINE_ROOT/pkgs" "$OFFLINE_ROOT/wheels"

cat > "$OFFLINE_ROOT/requirements_local.txt" <<'REQ'
torch==2.3.1+cpu
transformers==4.41.2
tokenizers==0.19.1
datasets==2.19.0
numpy==2.0.2
pandas==2.2.2
pyarrow==16.1.0
typing_extensions==4.12.2
sentencepiece==0.1.99
accelerate==0.31.0
safetensors==0.4.3
tqdm==4.66.4
REQ

# If the login node HAS internet:
python3.11 -m pip install -t "$OFFLINE_ROOT/pkgs" \
  --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r "$OFFLINE_ROOT/requirements_local.txt"

# If NO internet on login node:
python3.11 -m pip download -d "$OFFLINE_ROOT/wheels" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r "$OFFLINE_ROOT/requirements_local.txt"
python3.11 -m pip install -t "$OFFLINE_ROOT/pkgs" \
  --no-index --find-links "$OFFLINE_ROOT/wheels" \
  -r "$OFFLINE_ROOT/requirements_local.txt"
