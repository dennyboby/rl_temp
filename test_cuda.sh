#!/bin/bash

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 64G
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -p short
#SBATCH -t 20:00:00
#SBATCH --output=slurm_outputs/testjob_%j.out
#SBATCH --error=slurm_outputs/testjob_error_%j.out
#SBATCH -J den_rl_cuda

echo "RL Job running on $(hostname)"

echo "Loading Python Virtual Environment"

echo "Running Python Code"

python3 test_cuda.py