#!/bin/bash
#SBATCH --job-name=gnn-train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --partition=tue.gpu.q

# available partitions: CPU: tue.default.q , GPU: tue.gpu.q

mkdir -p ./data
mkdir -p ./logs

# Load modules
module load Python
module load CUDA/11.7.0
module load poetry/1.8.3-GCCcore-13.3.0

# Activate the Poetry environment
source $HOME/.cache/pypoetry/virtualenvs/dnr-gnn-_o2-hQZ5-py3.12/bin/activate

python --version 
cd $HOME/gnn-dnr
ls

python model_search/hpo.py --config ...