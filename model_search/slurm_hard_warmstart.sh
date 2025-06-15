#!/bin/bash
#SBATCH --job-name=exp_hard_all_gin
#SBATCH --output=logs/exp_hard_all_%A_%a.out
#SBATCH --error=logs/exp_hard_all_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH --partition=tue.default.q

# Create required directories
mkdir -p ./logs
mkdir -p ./results/hard_warmstart

echo "Loading modules..."
module purge
module load Python/3.11.3
module load poetry/1.5.1-GCCcore-12.3.0
module load Gurobi/11.0.3-GCCcore-12.3.0
echo "Modules loaded."

# Navigate to project directory
cd $HOME/gnn-dnr || exit 1

# Define all thresholds
THRESHOLDS=(0.9 0.7 0.5 0.3 0.1)
THRESHOLD=${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}

echo "Running Hard WarmStart with threshold: $THRESHOLD"

# Run experiment
poetry run python experiment_runner.py \
    --config_name GIN_HardWarmStart_thresh${THRESHOLD} \
    --model_path /vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/None-Best.pt \
    --model_config /vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/config-mlp.yaml \
    --test_folder /vast.mnt/home/20174047/gnn-dnr/data/split_datasets/test/ \
    --output_dir results/hard_warmstart