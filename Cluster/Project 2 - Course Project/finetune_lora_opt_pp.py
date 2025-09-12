#!/usr/bin/env python
# Fine-tune facebook/opt-2.7b with LoRA + DeepSpeed Pipeline Parallelism on CPU nodes.

import os, time, json, argparse, gc, math
import torch, torch.nn as nn, torch.distributed as dist
import deepspeed
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# --------------------------- Minimal, safe runtime knobs ---------------------------
# Use CPU-only DeepSpeed runtime; avoid building CUDA/cpp ops on shared nodes.
for v in ["DS_BUILD_OPS", "DS_SKIP_CXX_BUILD", "DEEPSPEED_DISABLE_SHM", "DEEPSPEED_USE_SHM"]:
    os.environ[v] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.expanduser("~/.cache/torch_extensions"))
try:
    # Keep thread usage tiny so 4 ranks fit on one 8-CPU node
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
except Exception:
    pass

# --------------------------- CLI ---------------------------
ap = argparse.ArgumentParser(description="LoRA finetune OPT-2.7B with DS pipeline parallel on CPU")
ap.add_argument("--model_name", default="facebook/opt-2.7b")
ap.add_argument("--data_file", required=True, help="plain-text file (one example per line)")
ap.add_argument("--seq_len", type=int, default=1024)   # 1024 fits CPU better than 2048
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--accum", type=int, default=64)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--logdir", default="logs")
ap.add_argument("--out_root", default=os.path.expanduser("~/finetuned"))
ap.add_argument("--ds_cfg", default="deepspeed_pp_zero1_cpu.json",
                help="Path to a DS json or leave to use an inline safe default")
args, _ = ap.parse_known_args()

# --------------------------- Distributed bootstrap (env://) ---------------------------
# Slurm + srun doesn't set MASTER_ADDR/PORT by default. We expect the launcher to export them.
rank  = int(os.environ.get("RANK", "0"))
world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
backend = "gloo"

if not dist.is_initialized():
    # Use env:// rendezvous (MASTER_ADDR/MASTER_PORT must be in environment)
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world)

def log(msg: str):
    print(f"[R{rank}] {msg}", flush=True)

# Number of pipeline stages = env var or world size (typical = 4)
STAGES = int(os.environ.get("PIPELINE_PARALLEL_SIZE", str(world)))
assert 1 <= STAGES <= 32, "PIPELINE_PARALLEL_SIZE must be in [1,32]"

# --------------------------- Tokenizer ---------------------------
tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

# --------------------------- Data ---------------------------
from datasets import load_dataset, load_from_disk

def _is_saved_dataset_dir(p: str) -> bool:
    return os.path.isdir(p) and os.path.exists(os.path.join(p, "dataset_info.json"))

if _is_saved_dataset_dir(args.data_file):
    print(f"[R{rank}] Loading pre-tokenized dataset from disk: {args.data_file}", flush=True)
    raw = load_from_disk(args.data_file)
else:
    print(f"[R{rank}] Loading RAW text file: {args.data_file}", flush=True)
    raw = load_dataset("text", data_files={"train": args.data_file})["train"]

# Shard by rank
if hasattr(raw, "shard"):
    raw = raw.shard(num_shards=world, index=rank, contiguous=True)

# If already tokenized (what you saved), just set format; otherwise tokenize now
cols = set(raw.features.keys())
if {"input_ids", "attention_mask"} <= cols:
    print(f"[R{rank}] Detected tokenized dataset (columns: {sorted(cols)})", flush=True)
    tok_ds = raw
    tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    print(f"[R{rank}] Detected raw text; tokenizing on the fly (seq_len={args.seq_len})", flush=True)
    def encode(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=args.seq_len)
    tok_ds = raw.map(encode, batched=True, remove_columns=["text"])
    tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])


# --------------------------- Base model + LoRA ---------------------------
# On CPU, float32 is safest (bf16 may exist but offers no speedup here).
base = AutoModelForCausalLM.from_pretrained(
    args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
)
base.config.use_cache = False
base.config.pad_token_id = tok.pad_token_id
try:
    base.gradient_checkpointing_enable()
except Exception:
    pass

peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(base, peft_cfg)

# --------------------------- Build Pipeline stages ---------------------------
# OPT decoder has N blocks. We'll split uniformly across STAGES.
opt_model  = peft_model.base_model.model.model if hasattr(peft_model, "base_model") else peft_model.model
decoder    = opt_model.decoder
dec_layers = decoder.layers
N = len(dec_layers)

class OPTEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = decoder.embed_tokens
        self.pos = decoder.embed_positions
    def forward(self, input_ids, attention_mask, labels=None, **_):
        # positions from mask (non-negative)
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        h = self.tok(input_ids) + self.pos(position_ids)
        return {"hidden_states": h, "attention_mask": attention_mask, "labels": labels}

class OPTBlk(nn.Module):
    def __init__(self, blk): super().__init__(); self.b = blk
    def forward(self, hidden_states, attention_mask, labels=None, **_):
        h, _ = self.b(hidden_states, attention_mask=attention_mask, use_cache=False)
        return {"hidden_states": h, "attention_mask": attention_mask, "labels": labels}

class OPTHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = decoder.final_layer_norm
        self.lm = peft_model.lm_head
        self.loss = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    def forward(self, hidden_states, labels=None, **_):
        logits = self.lm(self.ln(hidden_states))
        if labels is None:
            return {"logits": logits}
        # teacher-forcing next-token loss
        loss = self.loss(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
        )
        return {"loss": loss, "logits": logits}

# Split N blocks across STAGES uniformly
splits = [N // STAGES + (1 if i < N % STAGES else 0) for i in range(STAGES)]
stages = [OPTEmb()]
idx = 0
for n in splits:
    stages.append(nn.Sequential(*[OPTBlk(b) for b in dec_layers[idx:idx+n]]))
    idx += n
stages.append(OPTHead())

# Free large references early
del dec_layers, decoder, base
gc.collect()
log(f"Pipeline split={splits} (total blocks={N}, stages={STAGES})")

from deepspeed.pipe import PipelineModule
pipe = PipelineModule(
    layers=stages,
    num_stages=STAGES,
    loss_fn=None,
    partition_method="uniform",
)

# --------------------------- DeepSpeed engine ---------------------------
# Allow --ds_cfg to point to a JSON file; otherwise use an inline, CPU-safe config.
if isinstance(args.ds_cfg, str) and os.path.isfile(args.ds_cfg):
    ds_cfg_obj = args.ds_cfg
else:
    ds_cfg_obj = {
        "train_micro_batch_size_per_gpu": args.batch,
        "gradient_accumulation_steps": args.accum,
        "zero_optimization": {"stage": 1, "contiguous_gradients": True, "overlap_comm": False},
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},  # keep pure fp32 on CPUs
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0}
        },
        "gradient_clipping": 1.0,
        "pipeline": {"seed_layers": True}
    }

engine, _, _, _ = deepspeed.initialize(
    model=pipe,
    model_parameters=[p for p in pipe.parameters() if p.requires_grad],
    config=ds_cfg_obj,
    dist_init_required=False,  # we already called init_process_group(env://)
)
engine.set_lr(args.lr)
engine.set_gradient_accumulation_steps(args.accum)
my_stage = engine.grid.get_stage_id()
is_last  = (my_stage == STAGES - 1)
log(f"DeepSpeed initialized (stage {my_stage}/{STAGES})")

# --------------------------- Train ---------------------------
engine.train()
t0 = time.time()
for ep in range(args.epochs):
    for step, batch in enumerate(batch_iter(tok_ds, args.batch)):
        # DS Pipe expects tensors of type long
        for k in batch: batch[k] = batch[k].to(torch.long)
        loss = engine.train_batch(batch)
        if is_last and (step % 10 == 0):
            val = loss.item() if hasattr(loss, "item") else float(loss)
            log(f"ep={ep} step={step} loss={val:.4f} (+{time.time()-t0:.1f}s)")
            t0 = time.time()

# --------------------------- Save adapters (rank-0) ---------------------------
if engine.global_rank == 0:
    out = os.path.join(args.out_root, f"opt27b_lora_pp_{int(time.time())}")
    os.makedirs(out, exist_ok=True)
    # Save LoRA adapters + tokenizer (base weights come from HF)
    try:
        # some PEFT versions hang if saved with DS engine still alive; be explicit
        peft_model = engine.module if hasattr(engine, "module") else pipe
    except Exception:
        pass
    from peft import PeftModel
    # engine.module may wrap the PipelineModule; save the original PEFT model we created:
    # we kept 'peft_model' in closure above.
    peft_model.save_pretrained(out)
    tok.save_pretrained(out)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"split": splits, "stages": STAGES}, f, indent=2)
    log(f"Saved adapters+tokenizer to {out}")
