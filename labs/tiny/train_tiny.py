"""
Tiny (<1 M params) text classifier with HuggingFace Trainer + torchrun.
"""

import os, argparse
import torch                                   # already there – keep

# ───── Monkey-patch Accelerate for CPU-only nodes ─────
if not torch.cuda.is_available():              # no GPUs on the cluster
    try:
        import torch.cpu                       # <= exists in PyTorch 2.x
        torch.cpu.set_device = lambda *_: None # stub so Accelerate is happy
    except ImportError:
        pass
# ───────────────────────────────────────────────────────
from datasets import load_dataset
from transformers import (
    BertConfig, BertForSequenceClassification,
    AutoTokenizer, Trainer, TrainingArguments
)

def build_tokenizer(cache_root):
    # 8 k-token WordPiece vocab trained once & reused
    tok_name = "google/bert_uncased_L-2_H-128_A-2"   # use its 8 k vocab only
    return AutoTokenizer.from_pretrained(
        tok_name, cache_dir=os.path.join(cache_root, "tok")
    )

def tiny_config(vocab):
    return BertConfig(
        vocab_size=vocab,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=256,
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--subset", type=int, default=2000,  # tiny slice
                   help="training rows to keep")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default="./tiny_out")
    p.add_argument("--local_rank", type=int,
                   default=int(os.getenv("LOCAL_RANK", 0)))
    args = p.parse_args()

    # ------ distributed initialisation -----------------
    torch.distributed.init_process_group(
        backend="gloo" if not torch.cuda.is_available() else "nccl",
        rank=args.local_rank,
        world_size=int(os.getenv("WORLD_SIZE", 1)),
    )

    cache_root = os.environ.setdefault("HF_HOME",
                                       os.path.join(os.getcwd(), ".hf_cache"))
    os.makedirs(cache_root, exist_ok=True)

    ds = load_dataset("ag_news",
                      cache_dir=os.path.join(cache_root, "ds"))
    ds["train"] = ds["train"].select(range(args.subset))
    ds["test"]  = ds["test"].select(range(512))

    tok = build_tokenizer(cache_root)
    def tok_fn(b): return tok(b["text"], padding="max_length",
                              truncation=True, max_length=128)
    ds_tok = ds.map(tok_fn, batched=True).rename_column("label", "labels")
    ds_tok.set_format("torch",
                      columns=["input_ids", "attention_mask", "labels"])

    num_labels = ds_tok["train"].features["labels"].num_classes   # == 4
    # tell the config how many classes we have
    cfg  = tiny_config(tok.vocab_size)
    cfg.num_labels = num_labels        # <-- key line
    model = BertForSequenceClassification(cfg)  # no extra kwargs

    ta = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=5e-4,
        ddp_find_unused_parameters=False,
        logging_steps=10,
    )

    Trainer(model=model,
            args=ta,
            train_dataset=ds_tok["train"],
            eval_dataset=ds_tok["test"]).train()

if __name__ == "__main__":
    main()


'''
python - <<'PY'
import os, datasets, transformers, json
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/hf_tiny")
datasets.load_dataset("ag_news", split="train[:2000]",
                      cache_dir=f"{os.environ['HF_HOME']}/ds")
tok = transformers.AutoTokenizer.from_pretrained(
        "google/bert_uncased_L-2_H-128_A-2",
        cache_dir=f"{os.environ['HF_HOME']}/tok")
print("Done pre-caching.")
PY


salloc -N1 -n1 -c2 -p parallel --time=00:05:00 --exclusive

# 2) env + rendez-vous
source ~/llamaenv_local/bin/activate
cd ~/mrmito/project
export HF_HOME=$HOME/.cache/hf_tiny
export PYTHONPATH=$PWD:$PYTHONPATH
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((20000 + RANDOM % 10000))

torchrun \
  --nproc_per_node=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  labs/tiny/train_tiny.py \
  --subset 2000 --epochs 3
  
'''