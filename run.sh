#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=den_rl
#SBATCH -t 20:00:00
#SBATCH -C P100
#SBATCH --mem 80G
#SBATCH --gres=gpu:1
#SBATCH -p short

module load cuda10.2/toolkit/10.2.89
module load cudnn/8.1.1.33-11.2/3k5bbs63

source /home/dboby/rl_env/bin/activate

echo "Starting to run the code"
python main.py --train_dqn