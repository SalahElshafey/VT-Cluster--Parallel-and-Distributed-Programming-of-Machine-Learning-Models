# Fine-Tuning and Parallelization of Large Language Models on HPC Clusters

**Author:** Salaheldin Khaled Salaheldin Amin Elshafey  
**Email:** [Salahkhaledcx5@gmail.com](mailto:Salahkhaledcx5@gmail.com)  
**Date:** Summer 2025

---

## Overview

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
