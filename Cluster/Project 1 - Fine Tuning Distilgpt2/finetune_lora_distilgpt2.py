#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, time, json, socket, argparse, datetime, math, random

import torch
import torch.distributed as dist

# --- CLI ---
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["tiny","medium"], default="tiny")
ap.add_argument("--data_file", type=str, default="/data/tiny_openwebtext.txt")
ap.add_argument("--seq_len", type=int, default=256)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--accum", type=int, default=32)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--logdir", type=str, default="logs")
ap.add_argument("--out_root", type=str, default=os.path.expanduser("~/finetuned"))
args = ap.parse_args()

# --- CPU-only hygiene ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TORCH_DISTRIBUTED_BACKEND", "gloo")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch.cuda
torch.cuda.set_device = lambda *_: None

# Determinism-ish
torch.manual_seed(0); random.seed(0)

HOST = socket.gethostname()

# --- DDP init (torchrun provides env) ---
def ddp_init():
    if dist.is_initialized():
        return
    rank     = int(os.environ.get("RANK", "0"))
    world    = int(os.environ.get("WORLD_SIZE", "1"))
    master_a = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_p = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group(backend="gloo", init_method="env://")
    return rank, world, master_a, master_p

rank, world, master_a, master_p = ddp_init() or (0,1,"127.0.0.1","29500")
is_main = (rank == 0)
if is_main:
    print(f"[DDP] world={world} master={master_a}:{master_p}")
# --- DDP collective sanity ---
if dist.is_initialized():
    x = torch.ones(1)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    if is_main:
        print(f"[DDP] collective OK: sum={x.item()} world={world}", flush=True)

# --- Logging helper ---
os.makedirs(args.logdir, exist_ok=True)
def log_time(phase: str, seconds: float):
    p = os.path.join(args.logdir, f"timing_rank{rank}.log")
    with open(p, "a") as f:
        f.write(f"[{phase}] {seconds:.2f} sec\n")
    print(f"[{HOST}] rank={rank} ⏱ {phase}: {seconds:.2f}s", flush=True)

# fresh file
open(os.path.join(args.logdir, f"timing_rank{rank}.log"), "w").write("Timings (sec)\n")

# --- Data ---
from datasets import load_dataset
t0 = time.perf_counter()
dataset = load_dataset("text", data_files={"train": args.data_file})["train"]
log_time("Dataset load", time.perf_counter() - t0)

# --- Model / LoRA ---
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

BASE = "distilgpt2"
if is_main: print(f"[{HOST}] loading {BASE}")

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True
)
model.config.use_cache = False
model.config.pad_token_id = tok.pad_token_id
try:
    model.gradient_checkpointing_enable()
except Exception:
    pass

peft_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["c_attn","c_proj"],    # GPT-2 style attention proj
    lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_cfg)

# --- Tokenization ---
def encode(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=args.seq_len)

t0 = time.perf_counter()
tok_ds = dataset.map(encode, batched=True, remove_columns=["text"])
log_time("Tokenization", time.perf_counter() - t0)

collate = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# --- Trainer ---
from transformers import TrainerCallback
class StepTimer(TrainerCallback):
    def on_step_begin(self, argsT, state, control, **kw): self._t0 = time.perf_counter()
    def on_step_end(self, argsT, state, control, **kw):
        dt = time.perf_counter() - getattr(self, "_t0", time.perf_counter())
        print(f"[{HOST}] rank={rank} step {state.global_step} {dt:.3f}s", flush=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_name  = f"distilgpt2_lora_{args.dataset}_N{world}_{timestamp}"
save_dir  = os.path.join(args.out_root, run_name)
os.makedirs(save_dir, exist_ok=True)

t0 = time.perf_counter()
targs = TrainingArguments(
    output_dir=save_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=args.batch,
    gradient_accumulation_steps=args.accum,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    save_steps=500, save_total_limit=1,
    logging_steps=50,
    no_cuda=True, fp16=False, bf16=False,
    report_to="none",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    ddp_backend="gloo",
    dataloader_num_workers=2
)
trainer = Trainer(
    model=model,
    args=targs,
    train_dataset=tok_ds,
    tokenizer=tok,
    data_collator=collate,
    callbacks=[StepTimer()],
)
log_time("Trainer setup", time.perf_counter() - t0)

# --- Train ---
t0 = time.perf_counter()
trainer.train()
train_secs = time.perf_counter() - t0
log_time("Training", train_secs)

# --- Sync & Save (rank-0 only) ---
if dist.is_initialized(): dist.barrier()

t0 = time.perf_counter()
if trainer.is_world_process_zero():
    trainer.model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    meta = {
        "base_model": BASE,
        "world_size": world,
        "dataset": args.dataset,
        "data_file": args.data_file,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "batch": args.batch,
        "accum": args.accum,
        "lr": args.lr,
        "host": HOST,
        "train_seconds": train_secs
    }
    with open(os.path.join(save_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ saved adapter + tokenizer to {save_dir}", flush=True)
log_time("Model save", time.perf_counter() - t0)

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
