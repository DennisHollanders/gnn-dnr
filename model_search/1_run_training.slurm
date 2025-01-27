#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu

# Load modules
module purge
module load cuda/11.7

# Setup environment
export PYTHONPATH=$PWD:$PYTHONPATH
source $(poetry env info --path)/bin/activate

# Create required directories
mkdir -p ./data
mkdir -p ./logs

# Run training
python gnn_train.py