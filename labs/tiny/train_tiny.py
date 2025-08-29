"""
Tiny (<1 M params) text classifier with HuggingFace Trainer + torchrun.
Logs from EVERY rank to stdout (+ per-rank files) so Slurm tails show all nodes.
"""

import os
import argparse
import torch

# ── CPU-only monkey-patch so Accelerate won’t call torch.cpu.set_device() ──
if not torch.cuda.is_available():
    try:
        import torch.cpu  # type: ignore
        torch.cpu.set_device = lambda *_: None  # no-op on CPU-only nodes
    except Exception:
        pass
# ──────────────────────────────────────────────────────────────────────────

from datasets import load_dataset
from transformers import (
    BertConfig, BertForSequenceClassification,
    AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
)

# ---------- rank helpers ----------
def dist_is_init() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank() -> int:
    if dist_is_init():
        return torch.distributed.get_rank()
    return 0

def get_world_size() -> int:
    if dist_is_init():
        return torch.distributed.get_world_size()
    return 1

# ---------- per-rank logger (stdout + file, optional TB) ----------
class PerRankLogger(TrainerCallback):
    def __init__(self, out_dir: str, use_tb: bool = True):
        os.makedirs(out_dir, exist_ok=True)
        self.rank = get_rank()
        self.txt_path = os.path.join(out_dir, f"log.rank{self.rank}.txt")
        self._fh = open(self.txt_path, "a", buffering=1)  # line-buffered

        self.tb = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(out_dir, "tb", f"rank{self.rank}")
                os.makedirs(tb_dir, exist_ok=True)
                self.tb = SummaryWriter(tb_dir)
            except Exception:
                self.tb = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(state.global_step) if state.global_step is not None else -1
        prefix = f"[rank {self.rank} | step {step}] "
        line = prefix + " ".join(f"{k}={v}" for k, v in logs.items())
        # -> to stdout (captured by Slurm into bash.<jobid>.*.out)
        print(line, flush=True)
        # -> to per-rank text file
        self._fh.write(line + "\n")
        # -> TensorBoard (numeric only)
        if self.tb:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self._fh.close()
        except Exception:
            pass
        if self.tb:
            try:
                self.tb.flush(); self.tb.close()
            except Exception:
                pass

# ---------- helpers ----------
def build_tokenizer(cache_root: str):
    tok_name = "google/bert_uncased_L-2_H-128_A-2"  # tiny BERT vocab
    return AutoTokenizer.from_pretrained(tok_name,
                                         cache_dir=os.path.join(cache_root, "tok"))

def tiny_config(vocab_size: int) -> BertConfig:
    return BertConfig(
        vocab_size=vocab_size, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=256, max_position_embeddings=256,
    )

# ---------- main ----------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--subset", type=int, default=2000)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--out",    default="./tiny_out")
    p.add_argument("--local_rank", type=int,
                   default=int(os.getenv("LOCAL_RANK", 0)))
    args = p.parse_args()

    # DDP init so Trainer aggregates gradients across processes
    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    if not dist_is_init():
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    # Banner on EVERY rank (helps confirm all ranks alive in Slurm logs)
    if dist_is_init():
        print(f"[RANK {get_rank()}] WORLD_SIZE={get_world_size()}", flush=True)
    else:
        print("[single process]", flush=True)

    # Caches (works online or offline)
    cache_root = os.environ.setdefault("HF_HOME",
                                       os.path.join(os.getcwd(), ".hf_cache"))
    os.makedirs(cache_root, exist_ok=True)

    # ----- data
    ds = load_dataset("ag_news", cache_dir=os.path.join(cache_root, "ds"))
    ds["train"] = ds["train"].select(range(args.subset))
    ds["test"]  = ds["test"].select(range(512))

    tok = build_tokenizer(cache_root)
    def tok_fn(batch):
        return tok(batch["text"], padding="max_length",
                   truncation=True, max_length=128)
    ds_tok = ds.map(tok_fn, batched=True).rename_column("label", "labels")
    ds_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # ----- model
    cfg = tiny_config(tok.vocab_size)
    cfg.num_labels = ds_tok["train"].features["labels"].num_classes  # = 4
    model = BertForSequenceClassification(cfg)

    # ----- training args (ensure logging happens periodically on all ranks)
    rank = get_rank()
    tr_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=5e-4,
        ddp_find_unused_parameters=False,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=(rank != 0),   # keep progress bar only on rank 0
        report_to=[],               # no external trackers by default
    )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
    )

    # Attach per-rank logger so EVERY rank emits logs
    trainer.add_callback(PerRankLogger(args.out, use_tb=True))

    # Train
    trainer.train()

    # Save once (rank 0)
    if not dist_is_init() or get_rank() == 0:
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()
