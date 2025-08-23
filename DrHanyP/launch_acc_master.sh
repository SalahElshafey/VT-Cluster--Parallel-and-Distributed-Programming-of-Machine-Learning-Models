#!/bin/bash
nodes=(hpc11 hpc12 hpc15 hpc16 hpc18 hpc21 hpc25 hpc26 hpc27 hpc29 hpc32 hpc35 hpc36 hpc37 hpc39 hpc40 hpc41 hpc43 hpc44 hpc46)

for i in "${!nodes[@]}"; do
  node="${nodes[$i]}"
  echo "Launching on $node with RANK $i"
  ssh "$node" "
    cd ~/Summer-Training-2025/Dgpt2/distilgpt2-finetune
    RANK=${i} bash launch_acc.sh
  " &
done
