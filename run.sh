#!/bin/bash

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 64G
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -p short
#SBATCH -t 20:00:00
#SBATCH -o ~/dboby/rl_temp/slurm_outputs/rljob_%j.out
#SBATCH --error ~/dboby/rl_temp/slurm_outputs/rljob_error_%j.out
#SBATCH -J den_rl

echo "RL Job running on $(hostname)"

echo "Loading Python Virtual Environment"

source ~/dboby/rl_env/bin/activate

echo "Running Python Code"

python3 ~/dboby/rl_temp/main.py --train_dqn