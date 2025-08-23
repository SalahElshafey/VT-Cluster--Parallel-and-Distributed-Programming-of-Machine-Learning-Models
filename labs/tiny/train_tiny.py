"""
Tiny (<1 M params) text classifier with HuggingFace Trainer + torchrun.
"""

"""
Tiny (<1 M params) text classifier with HuggingFace Trainer + torchrun.
"""

import os, argparse, torch

# ── CPU-only monkey-patch so Accelerate won’t call torch.cpu.set_device() ──
if not torch.cuda.is_available():
    try:
        import torch.cpu
        torch.cpu.set_device = lambda *_: None
    except ImportError:
        pass
# ──────────────────────────────────────────────────────────────────────────
from datasets import load_dataset
from transformers import (
    BertConfig, BertForSequenceClassification,
    AutoTokenizer, Trainer, TrainingArguments,
)

# ---------- helpers -------------------------------------------------------
def build_tokenizer(cache_root):
    tok_name = "google/bert_uncased_L-2_H-128_A-2"      # 8 k-token vocab
    return AutoTokenizer.from_pretrained(tok_name,
                                         cache_dir=os.path.join(cache_root, "tok"))

def tiny_config(vocab_size: int) -> BertConfig:
    return BertConfig(
        vocab_size=vocab_size, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=256, max_position_embeddings=256,
    )

# ---------- main ----------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--subset", type=int, default=2000)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--out",    default="./tiny_out")
    p.add_argument("--local_rank", type=int,
                   default=int(os.getenv("LOCAL_RANK", 0)))
    args = p.parse_args()

    # Initialise torch.distributed so Trainer aggregates gradients
# ── add this instead ──
    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    # 🔎 add debug print here
    if torch.distributed.is_initialized():
        print(f"[RANK {torch.distributed.get_rank()}] "
              f"WORLD_SIZE={torch.distributed.get_world_size()}",
              flush=True)
    else:
        print("[single process]", flush=True)
    cache_root = os.environ.setdefault("HF_HOME",
                                       os.path.join(os.getcwd(), ".hf_cache"))
    os.makedirs(cache_root, exist_ok=True)

    # ---------- data ----------
    ds = load_dataset("ag_news", cache_dir=os.path.join(cache_root, "ds"))
    ds["train"] = ds["train"].select(range(args.subset))
    ds["test"]  = ds["test"].select(range(512))

    tok = build_tokenizer(cache_root)
    def tok_fn(b): return tok(b["text"], padding="max_length",
                              truncation=True, max_length=128)
    ds_tok = ds.map(tok_fn, batched=True).rename_column("label", "labels")
    ds_tok.set_format("torch",
                      columns=["input_ids", "attention_mask", "labels"])

    # ---------- model ----------
    cfg = tiny_config(tok.vocab_size)
    cfg.num_labels = ds_tok["train"].features["labels"].num_classes  # =4
    model = BertForSequenceClassification(cfg)

    # ---------- training ----------
    tr_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=5e-4, # Dynamic learning rate.
        ddp_find_unused_parameters=False,
        logging_steps=10,
    )

    trainer = Trainer(model=model,
                      args=tr_args,
                      train_dataset=ds_tok["train"],
                      eval_dataset=ds_tok["test"])
    trainer.train() # model under training

    # 🔑 NEW: save model + tokenizer *once* (rank 0)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()

