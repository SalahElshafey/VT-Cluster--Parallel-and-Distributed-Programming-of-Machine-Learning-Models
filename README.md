Absolutely. Here's an updated and polished version of your `README.md` — tailored to be \*\*professional\*\*, \*\*realistic\*\*, and \*\*transparent\*\* about the cluster access.



This version includes a clear and respectful note indicating that the HPC cluster is \*\*restricted to authorized users\*\*, and the scripts are provided for demonstration, documentation, and portfolio purposes.



---



````markdown

\# Fine-Tuning and Parallelization of Large Language Models on HPC Clusters



\*\*Author:\*\* Salaheldin Khaled Salaheldin Amin Elshafey  

\*\*Email:\*\* \[Salahkhaledcx5@gmail.com](mailto:Salahkhaledcx5@gmail.com)  

\*\*Date:\*\* Summer 2025



---



\## Overview



This repository documents two advanced machine learning projects that demonstrate scalable fine-tuning of Large Language Models (LLMs) using a distributed high-performance computing (HPC) cluster.



The work focuses on implementing parallelization strategies such as \*\*Distributed Data Parallel (DDP)\*\*, \*\*Model Sharding\*\*, \*\*Pipeline Parallelism\*\*, and \*\*Low-Rank Adaptation (LoRA)\*\* to make large-scale training feasible under constrained hardware environments.



The models used:



\- \[`distilgpt2`](https://huggingface.co/distilgpt2): A compact GPT-2 variant (~82M parameters).

\- \[`facebook/opt-2.7b`](https://huggingface.co/facebook/opt-2.7b): A large-scale decoder-only transformer (~2.7B parameters).



---



\## ⚠️ Note on Cluster Access



> The scripts and benchmarks in this repository were developed and tested on a private multi-node HPC cluster.  

> \*\*Access to the cluster is restricted\*\* to authorized users only.  

> This repository is provided for \*\*documentation, demonstration, and academic portfolio purposes\*\*.



All included scripts and configurations reflect actual training procedures, parallelization strategies, and optimization experiments conducted in a real cluster environment.



---



\## Project 1: Fine-Tuning DistilGPT2 with DDP and LoRA



\### Summary



This project fine-tunes `distilgpt2` using:



\- \*\*Distributed Data Parallel (DDP)\*\* for multi-node execution

\- \*\*LoRA (Low-Rank Adaptation)\*\* for memory-efficient fine-tuning



\### Technical Details



\- Model: Decoder-only Transformer

\- Parameters: ~82M

\- Layers: 6

\- Hidden Size: 768

\- Attention Heads: 12

\- Context Length: 1024 tokens

\- Tokenizer: GPT-2 BPE

\- Optimizations: DDP, LoRA



\### Datasets



\- Debug Dataset: `/data/tiny\_openwebtext.txt`

\- Medium Dataset (~20,000 lines): `/data/medium\_openwebtext.txt`



\### Cluster Configuration



Example of nodes used:



```bash

nodes=(hpc11 hpc12 hpc15 hpc16 hpc18 hpc21 hpc25 hpc26 hpc27 hpc29 \\

&nbsp;      hpc32 hpc35 hpc36 hpc37 hpc39 hpc40 hpc41 hpc43 hpc44 hpc46)

````



\### Benchmarking



Training was executed on:



\* 5 nodes

\* 10 nodes

\* 20 nodes



Training times and performance metrics were logged and compared.

Output models are saved under: `~/distilgpt2\_finetuned/`



---



\## Project 2: Fine-Tuning OPT-2.7B Using Parallelism and Memory Optimization



\### Summary



This project scales the fine-tuning of `facebook/opt-2.7b`, a 2.7 billion parameter model, across a 40-node CPU cluster. Due to limited memory (8 GB per node), DDP alone was insufficient. A hybrid approach using \*\*Model Parallelism\*\*, \*\*Pipeline Parallelism\*\*, \*\*LoRA\*\*, and \*\*FP16 Mixed Precision\*\* was implemented.



\### Model Specifications



\* Architecture: Decoder-only Transformer

\* Layers: 32

\* Hidden Size: 2560

\* Attention Heads: 32

\* Feedforward Network Size: 10240

\* Activation: ReLU

\* Tokenizer: GPT-2 BPE (vocab size: 50,272)

\* Context Window: 2048 tokens



\### Techniques Applied



\* Partitioned the model using frameworks like DeepSpeed or Megatron-LM

\* Distributed layers across multiple nodes using pipeline parallelism

\* Applied LoRA to reduce memory usage during fine-tuning

\* Enabled FP16 for reduced memory footprint

\* Sharded dataset and managed cross-node synchronization



\### Dataset



\* Medium dataset used: `/data/medium\_openwebtext.txt`



\### Results



\* Successfully trained across 40 nodes

\* Memory constraints were addressed through careful partitioning

\* Checkpoints saved to: `~/opt\_2.7b\_finetuned/`

\* Final performance analysis is included in the `benchmarks/` directory



---



\## Installation



```bash

\# Clone the repository

git clone https://github.com/your-username/llm-parallel-training.git

cd llm-parallel-training



\# Set up Python virtual environment

python3 -m venv env

source env/bin/activate



\# Install all required dependencies

pip install -r requirements.txt

```



---



\## Usage



\### Fine-Tune DistilGPT2



```bash

\# Debug run (tiny dataset)

bash scripts/train\_distilgpt2.sh tiny



\# Full benchmark run (medium dataset)

bash scripts/train\_distilgpt2.sh medium

```



\### Fine-Tune OPT-2.7B



```bash

\# Full parallel training using 40-node strategy

bash scripts/train\_opt\_2.7b.sh

```



> Note: These scripts are intended for clusters with SLURM or MPI-based job scheduling.



---



\## Dependencies



This project uses the following Python packages:



\* `transformers`

\* `datasets`

\* `torch`

\* `accelerate`

\* `deepspeed`

\* `peft`

\* `tokenizers`

\* `numpy`, `scipy`, `pandas`



For a complete list, refer to the `requirements.txt` file.



---



\## System Requirements



\* Access to a multi-node HPC cluster

\* Each node with:



&nbsp; \* At least 8 GB RAM

&nbsp; \* Linux OS

\* Shared file system (e.g., NFS or Lustre)

\* Python 3.8 or higher

\* SLURM or MPI-compatible job launcher



---



\## Troubleshooting



| Issue                    | Solution                                                                 |

| ------------------------ | ------------------------------------------------------------------------ |

| Out of Memory (OOM)      | Use LoRA with pipeline or model parallelism                              |

| NCCL Initialization Fail | Verify `MASTER\_ADDR`, `RANK`, `WORLD\_SIZE` environment variables         |

| Tokenizer Errors         | Use the correct Hugging Face tokenizer class (e.g., `GPT2TokenizerFast`) |

| I/O Bottlenecks          | Use node-local storage or `/scratch` if available                        |



---



\## Author



\*\*Salaheldin Khaled Salaheldin Amin Elshafey\*\*

Email: \[Salahkhaledcx5@gmail.com](mailto:Salahkhaledcx5@gmail.com)



---



\## License



This project is distributed under the \[MIT License](LICENSE).



---



```



---



\### ✅ Summary



This version:



\- ✅ Clearly separates your work from the system/infrastructure access

\- ✅ Is professional and understated

\- ✅ Highlights your technical work, not just the code

\- ✅ Prepares you for applying to research internships, jobs, or master's/PhD programs



Let me know if you want:



\- A matching `requirements.txt`

\- Example SLURM scripts (`sbatch`) or launchers

\- A professional GitHub project description or LinkedIn post



You're now fully ready to publish this as a \*\*credible, technical, and personal\*\* achievement.

```



