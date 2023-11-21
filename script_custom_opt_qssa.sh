#!/bin/bash -l
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --job-name Custom_Objective
#SBATCH --ntasks-per-node=1
#SBATCH --output outputs/custom_objective.o%j
#SBATCH --time=3:00:00
source ~/.bashrc
srun --tasks=1 --cpus-per-task=40 /home/htc/tokunola/miniconda3/envs/pesto_new/bin/python custom_optimization_qssa.py
