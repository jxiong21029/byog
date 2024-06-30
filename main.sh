#!/bin/bash
#SBATCH -p cvmlp-lab
#SBATCH -c 60
#SBATCH --output "/coc/testnvme/jxiong60/vpt-llm/byog/runs/%j/stdout.log"
#SBATCH --error "/coc/testnvme/jxiong60/vpt-llm/byog/runs/%j/stderr.log"
#SBATCH --gpus-per-node a40:4
#SBATCH --qos "short"
#SBATCH --exclude qt-1

export PYTHONUNBUFFERED=TRUE
cd /coc/testnvme/jxiong60/vpt-llm

srun -u python -u -m accelerate.commands.launch --main_process_port 29501 byog/main.py
