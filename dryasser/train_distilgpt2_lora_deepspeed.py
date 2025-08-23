import os
os.environ["DS_BUILD_OPS"] = "0"  # 👈 THIS is the critical missing line
os.environ["TORCH_DISTRIBUTED_BACKEND"] = "gloo"
os.environ["ACCELERATE_DISABLE"] = "false"
os.environ["PATH"] += ":/home/yhanafy/.local/bin"
os.environ["CUDA_VISIBLE_DEVICES"] = " "
os.environ["ACCELERATE_USE_CPU"] = "true"
os.environ["USE_CPU_ONLY"] = "true"
os.environ["DS_SKIP_CUDA_CHECK"] = "1"

import socket

print(f"\n[{socket.gethostname()}] 🛠 ENV DEBUG")
print(f"  TORCH_DISTRIBUTED_BACKEND = {os.environ.get('TORCH_DISTRIBUTED_BACKEND')}")
print(f"  CUDA_VISIBLE_DEVICES      = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"  ACCELERATE_DISABLE        = {os.environ.get('ACCELERATE_DISABLE')}")
print(f"  ACCELERATE_USE_CPU        = {os.environ.get('ACCELERATE_USE_CPU')}")
print(f"  USE_CPU_ONLY              = {os.environ.get('USE_CPU_ONLY')}")
print(f"  DEEPSPEED_BUILD_OPS       = {os.environ.get('DS_BUILD_OPS')}")
print(f"  PYTHONPATH                = {os.environ.get('PYTHONPATH')}")
print(f"  DS_BUILD_OPS              = {os.environ.get('DS_BUILD_OPS')}")
print(f"  Host                      = {socket.gethostname()}")
print(f"  PID                       = {os.getpid()}")
print(f"  WORLD SIZE                = {os.environ.get('WORLD_SIZE')}")
print("-" * 60 + "\n")

import torch
import time
import socket
hostname = socket.gethostname()
rank = int(os.environ.get("RANK", 0))

# Monkey-patch torch.cuda.set_device to no-op (to prevent crash on CPU-only nodes)
import torch.cuda
torch.cuda.set_device = lambda *_: None

import torch.distributed
print(f"[{socket.gethostname()}] Distributed initialized: {torch.distributed.is_initialized()}")
print(f"[{socket.gethostname()}] RANK = {os.environ.get('RANK')}, WORLD_SIZE = {os.environ.get('WORLD_SIZE')}")

print(f" CUDA available =========== {torch.cuda.is_available()}")

# ==================== adjust the time out of the network to a slow network ======

from datetime import timedelta
import torch.distributed as dist

hostname = socket.gethostname()
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

if dist.is_available() and not dist.is_initialized():
    dist.init_process_group(
        backend="gloo",  # or "nccl" if using GPU
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=600)  # ✅ Increase timeout to 5 minutes
    )
    print(f"[{hostname}] ✅ DDP initialized: RANK={rank}, WORLD_SIZE={world_size}")


# ============ Helper debugging function to print in a log file =============

def log_timing(phase, duration):
    try:
        log_path = f"timing_rank{rank}.log"
        with open(log_path, "a") as f:
            f.write(f"[{phase}] {duration:.2f} sec\n")
            f.flush()            # Ensure Python flushes its buffer
            os.fsync(f.fileno()) # Ensure OS flushes to disk
        print(f"[{hostname}] ✅ Logged '{phase}' = {duration:.2f} sec to {log_path}")
    except Exception as e:
        print(f"[{hostname}] ❌ Failed to log timing for '{phase}': {e}")

with open(f"timing_rank{rank}.log", "w") as f:
        f.write(f" Timings in sec\n")


# ==========Sync ALL nodes ============
from torch.distributed import barrier

if torch.distributed.is_initialized():
    barrier()  # All nodes sync here before timing training
start = time.time()
# ========= start program ============
# print(f"[{hostname}] RANK={rank} starting point took  {time.time() - start:.2f} sec") 

# === Load dataset ===
from datasets import load_dataset
start = time.time()
dataset = load_dataset("text", data_files={"train": "/data/medium_openwebtext.txt"})["train"]
# print(f"[{hostname}] RANK={rank} dataset loaded in {time.time() - start:.2f} sec")
log_timing("Dataset load", time.time() - start)

# === Tokenizer & Model ===
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# === Apply LoRA ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)

# === Tokenize Dataset ===
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

start = time.time()
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
# print(f"[{hostname}] RANK={rank} tokenization took {time.time() - start:.2f} sec")
log_timing("Tokenization", time.time() - start)

# === Data Collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Training Arguments ===
output_dir = "./ds_distilgpt2_lora_deepspeed"
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    save_steps=500,
    save_total_limit=1,
    logging_steps=50,
    learning_rate=5e-5,
    fp16=False,
    push_to_hub=False,
    report_to="none",
    remove_unused_columns=False,
    deepspeed="./ds_config_cpu.json",
    ddp_find_unused_parameters=False,
    ddp_backend="gloo",
)

# ============== debugging function to compute the step time ==========

from transformers import TrainerCallback
import time

class StepTimerCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.start_time
        print(f"[{socket.gethostname()}] Step {state.global_step} took {step_time:.3f} sec")

# === Trainer ===
start = time.time()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[StepTimerCallback()],
)
# print(f"[{hostname}] RANK={rank} trainer setup took {time.time() - start:.2f} sec")
log_timing("Trainer setup ", time.time() - start)

# Trainer is already created at this point
train_loader = trainer.get_train_dataloader()

# =========  DEBUGGING CODE Grab the sampler
sampler = train_loader.sampler

import socket
hostname = socket.gethostname()
rank = int(os.environ.get("RANK", 0))

if hasattr(sampler, 'indices'):
    # For RandomSampler or SequentialSampler
    indices = sampler.indices
elif hasattr(sampler, 'num_replicas'):
    # For DistributedSampler
    indices = list(sampler.__iter__())[:10]  # Show first 10
    print(f"[{hostname}] RANK={rank} sees {len(sampler)} samples")
else:
    indices = None

if indices is not None:
    print(f"[{hostname}] RANK={rank} sample indices preview: {indices[:10]}")
else:
    print(f"[{hostname}] RANK={rank} could not extract sample indices.")
#  ========  END DEBUGGING CODE 


if torch.distributed.is_initialized():
    barrier()  # All nodes sync here before training
print(f"[{hostname}] RANK={rank} eached the second BARRIER")

# === Train ===
start = time.time()
trainer.train()
# print(f"[{hostname}] RANK={rank} training took {time.time() - start:.2f} sec")

log_timing("Training ", time.time() - start)

# === Save (main process only) ===
from transformers.trainer_utils import is_main_process
start = time.time()

if is_main_process(training_args.local_rank):
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to {output_dir}")
# print(f"[{hostname}] RANK={rank} model save took {time.time() - start:.2f} sec")
log_timing("Model Save ", time.time() - start)

