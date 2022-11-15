#!/bin/bash

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 64G
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -p short
#SBATCH -t 20:00:00
#SBATCH --output=slurm_outputs/rljob_%j.out
#SBATCH --error=slurm_outputs/rljob_error_%j.out
#SBATCH -J den_rl

echo "RL Job running on $(hostname)"

echo "Loading Python Virtual Environment"

echo "Running Python Code"

python3 main.py --train_dqn