#!/bin/bash
#SBATCH -J unet
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o unet.out

source "/gs/software/anaconda3/bin/activate" dlenv

python3 /gs/home/majg/liupeng/unet/main.py train

sleep 200
