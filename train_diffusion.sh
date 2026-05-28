#!/bin/bash

#MSUB -r diffusion_train
#MSUB -q a100
#MSUB -n 1
#MSUB -c 64
#MSUB -T 86400
#MSUB -m work,scratch
#MSUB -o logs/out_%I.txt
#MSUB -e logs/err_%I.txt

set -x

cd $CCCWORKDIR/diffusion

module purge
module load python3
module load cuda

source ../.venv/bin/activate

nvidia-smi

python diffusion.py
