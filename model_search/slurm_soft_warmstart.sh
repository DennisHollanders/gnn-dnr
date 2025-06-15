#!/bin/bash
#SBATCH --job-name=exp_soft_gin
#SBATCH --output=logs/exp_soft_%A_%a.out
#SBATCH --error=logs/exp_soft_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --partition=tue.default.q

# Create required directories
mkdir -p ./logs
mkdir -p ./results/soft_warmstart

echo "Loading modules..."
module purge
module load Python/3.11.3
module load poetry/1.5.1-GCCcore-12.3.0
module load Gurobi/11.0.3-GCCcore-12.3.0
echo "Modules loaded."

# Navigate to project directory
cd $HOME/gnn-dnr || exit 1

# Define experiments
EXPERIMENTS=(
    "GIN_SoftWarmStart_Floats"
    "GIN_SoftWarmStart_BinaryRounding"
    "GIN_SoftWarmStart_BinaryPhyR"
)

# Get experiment for this array task
EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "Running experiment: $EXPERIMENT"

# Create config file for this experiment
poetry run python -c "
import json
from experiment_runner import ExperimentConfig

configs = {
    'GIN_SoftWarmStart_Floats': {'warmstart_mode': 'float'},
    'GIN_SoftWarmStart_BinaryRounding': {'warmstart_mode': 'soft', 'use_rounding': True},
    'GIN_SoftWarmStart_BinaryPhyR': {'warmstart_mode': 'soft', 'use_phyr': True}
}

config = ExperimentConfig('GIN', 'SoftWarmStart', '${EXPERIMENT}'.split('_')[-1], configs['${EXPERIMENT}'])
with open('config_${SLURM_ARRAY_TASK_ID}.json', 'w') as f:
    json.dump(config.to_dict(), f)
"

# Run experiment
poetry run python experiment_runner.py \
    --config_file config_${SLURM_ARRAY_TASK_ID}.json \
    --model_path /vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/None-Best.pt \
    --model_config /vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/config-mlp.yaml \
    --test_folder /vast.mnt/home/20174047/gnn-dnr/data/split_datasets/test/ \
    --output_dir results/soft_warmstart

# Cleanup
rm config_${SLURM_ARRAY_TASK_ID}.json