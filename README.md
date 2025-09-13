# Fine-Tuning and Parallelization of Large Language Models on HPC Clusters

<p align="center">
<img src="assets/Virginia-Tech-Logo.png" alt="Virginia Tech Logo" height="150">
&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/AAST.png" alt="AASTMT Logo" height="150">
    
</p>

**Author:** Salaheldin Khaled Salaheldin Amin Elshafey  
**Email:** [Salahkhaledcx5@gmail.com](mailto:Salahkhaledcx5@gmail.com)  
**Date:** Summer 2025

---

## Overview


This repository documents two advanced machine learning projects that demonstrate scalable fine-tuning of Large Language Models (LLMs) on a multi-node high-performance computing (HPC) cluster with limited memory per node.

The work implements distributed training and memory-efficient techniques such as:

- **Distributed Data Parallel (DDP)**
- **Model Parallelism**
- **Pipeline Parallelism**
- **Low-Rank Adaptation (LoRA)**
- **FP16 Mixed Precision Training**

The goal was to adapt large-scale models to hardware-constrained environments and evaluate their performance across different parallelization strategies.

Models fine-tuned:

- [`distilgpt2`](https://huggingface.co/distilgpt2): Compact GPT-2 variant (~82M parameters)
- [`facebook/opt-2.7b`](https://huggingface.co/facebook/opt-2.7b): Large decoder-only transformer (~2.7B parameters)

---

## ⚠️ Note on Cluster Access

> These experiments were conducted on a private, multi-node CPU-based HPC cluster.  
> **Access is restricted to authorized users only.**  
> This repository is shared for documentation and academic showcase purposes.

---

## Project 1: Fine-Tuning DistilGPT2 with DDP and LoRA

### Objective

To fine-tune the lightweight `distilgpt2` model using **Distributed Data Parallel (DDP)** and **LoRA**, and evaluate execution performance across multiple cluster node configurations.

### Model Specifications

- **Architecture:** GPT-2–style causal LM (decoder-only Transformer)
- **Parameters:** ~82M
- **Layers / Heads / Hidden Size:** 6 layers, 12 attention heads, 768 hidden units
- **Context Length:** 1024 tokens
- **Tokenizer:** GPT-2 byte-level BPE (50,257 vocab); EOS/BOS token ID: 50256
- **Dropout:** attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop=0.1
- **Activation Function:** `gelu_new`
- **License:** Apache-2.0

### Datasets

- **Tiny Dataset for Debugging:** `/data/tiny_openwebtext.txt`
- **Medium Dataset (~20,000 lines):** `/data/medium_openwebtext.txt`

### Cluster Configuration

Training was executed on a cluster with 8GB RAM per node. Nodes used:

```bash
nodes=(hpc11 hpc12 hpc15 hpc16 hpc18 hpc21 hpc25 hpc26 hpc27 hpc29 \
       hpc32 hpc35 hpc36 hpc37 hpc39 hpc40 hpc41 hpc43 hpc44 hpc46)
```

### Execution and Benchmarking

- Tiny dataset used for testing/debugging
- Medium dataset used for final benchmarks
- Model trained using:
  - 5 nodes
  - 10 nodes
  - 20 nodes
- Execution time measured and compared
- Final model saved in: `~/distilgpt2_finetuned/`

---

## Project 2: Fine-Tuning facebook/opt-2.7b with Parallelism & Memory Optimization

### Objective

To fine-tune `facebook/opt-2.7b`, a large-scale decoder-only transformer, on a 40-node CPU cluster using optimized memory distribution and parallelism.

Due to model size and memory constraints (each node has only 8 GB RAM), standard DDP is not feasible. Instead, this project used:

- **Model Sharding**
- **Pipeline Parallelism**
- **LoRA**
- **FP16 mixed precision**

### Model Specifications

- **Type:** Decoder-only Transformer (causal LM)
- **Parameters:** ~2.7B  
  - FP16: ≈ 5.4 GB  
  - FP32: ≈ 10.8 GB
- **Layers / Attention Heads / Hidden Size:** 32 layers, 32 heads, 2560 hidden units
- **Feedforward Network Size:** 10,240
- **Activation Function:** ReLU
- **LayerNorm Placement:** Pre-LN (`do_layer_norm_before: true`)
- **Dropout:**
  - residual: 0.1  
  - attention: 0.0  
  - activation: 0.0
- **Context Length:** 2048 tokens
- **Tokenizer & Vocab:** GPT-2 byte-level BPE, vocab size: 50,272  
  - `pad_token_id=1`, `bos_token_id=2`, `eos_token_id=2`
- **HF Architecture Class:** `OPTForCausalLM`
- **Training Objective:** Causal Language Modeling (trained on ~180B tokens from curated corpora)

### Datasets

- Reused the medium-sized dataset: `/data/medium_openwebtext.txt`

### Execution Strategy

- **Why not DDP?**: Model exceeds per-node memory
- **Solution**:
  - Model sharded across layers using pipeline parallelism
  - LoRA used to avoid full fine-tuning
  - Training done in FP16 to reduce memory consumption
  - Dataset sharded manually per node

### Output

- Training completed using 40 nodes
- Checkpoints saved in: `~/opt_2.7b_finetuned/`
- Benchmarking report includes node utilization, memory stats, and training times

---

## Installation

```bash
# Clone the repository
git clone https://github.com/AIBabyTeaching/Cluster.git
cd project/Cluster

# Set up Python environment
python3 -m venv env
source source ~lamaenv_local/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Fine-tune DistilGPT2

```bash
#Get to workdir
cd 'Project 1 - Fine Tuning Distilgpt2'
# Debug run 
bash submit_distilgpt2_lora.sbatch

```

### Fine-tune OPT-2.7B

```bash
#Get to workdir
cd 'Project 2 - Course Project'
# Distributed fine-tuning with model/pipeline parallelism
bash submit_opt27b_pp.sbatch
```

> Scripts assume SLURM for node orchestration.

---

## Dependencies

Key Python libraries used:

- `transformers==4.53.2`
- `datasets==4.0.0`
- `torch==2.1.0`
- `accelerate==1.9.0`
- `tokenizers==0.21.2`
- `numpy==1.26.4`, `scipy==1.16.0`, `pandas==2.3.1`

See `requirements.txt` for exact versions.

---

## System Requirements

- Multi-node HPC cluster with:
  - 5 to 40 nodes (depending on the task)
  - Each node with minimum 8 GB RAM
- Linux OS
- Shared file system (e.g., `/data/`)
- SLURM job launcher
- Python 3.8+

---

## Troubleshooting

| Issue                  | Suggested Fix                                                         |
|------------------------|------------------------------------------------------------------------|
| Out of Memory (OOM)    | Use LoRA + pipeline/model sharding + FP16 precision                   |
| NCCL Launch Errors     | Confirm `MASTER_ADDR`, `MASTER_PORT`, `RANK`, and `WORLD_SIZE` setup  |
| Tokenizer Mismatch     | Ensure use of correct tokenizer class (e.g., `GPT2TokenizerFast`)      |
| Slow I/O or loading    | Prefer node-local storage or `/scratch` instead of NFS                |

---

## Author

**Salaheldin Khaled Salaheldin Amin Elshafey**  
Email: [Salahkhaledcx5@gmail.com](mailto:Salahkhaledcx5@gmail.com)

---

## License
This project is licensed under the [MIT License](LICENSE).

This repository documents two advanced machine learning projects that demonstrate scalable fine-tuning of Large Language Models (LLMs) using a distributed high-performance computing (HPC) cluster.

The work focuses on implementing parallelization strategies such as **Distributed Data Parallel (DDP)**, **Model Sharding**, **Pipeline Parallelism**, and **Low-Rank Adaptation (LoRA)** to make large-scale training feasible under constrained hardware environments.

The models used:

- [`distilgpt2`](https://huggingface.co/distilgpt2): A compact GPT-2 variant (~82M parameters).
- [`facebook/opt-2.7b`](https://huggingface.co/facebook/opt-2.7b): A large-scale decoder-only transformer (~2.7B parameters).

---

## ⚠️ Note on Cluster Access

> The scripts and benchmarks in this repository were developed and tested on a private multi-node HPC cluster.  
> **Access to the cluster is restricted** to authorized users only.  
> This repository is provided for **documentation, demonstration, and academic portfolio purposes**.

