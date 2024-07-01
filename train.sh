#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=train_output.log
#SBATCH --error=train_error.log

module load anaconda-py3/2021.05
source activate Colin1
python train.py
conda deactivate