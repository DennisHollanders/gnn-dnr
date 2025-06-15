#!/bin/bash
#SBATCH --job-name=exp_direct_gin
#SBATCH --output=logs/exp_direct_%A_%a.out
#SBATCH --error=logs/exp_direct_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-1
#SBATCH --partition=tue.default.q

# Create required directories
mkdir -p ./logs
mkdir -p ./results/direct_prediction

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
    "GIN_DirectPrediction_Rounding"
    "GIN_DirectPrediction_PhyR"
)

# Get experiment for this array task
EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "Running experiment: $EXPERIMENT"

# Run experiment using poetry
poetry run python experiment_runner.py \
    --config_name $EXPERIMENT \
    --model_path /vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/None-Best.pt \
    --model_config /vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/config-mlp.yaml \
    --test_folder /vast.mnt/home/20174047/gnn-dnr/data/split_datasets/test/ \
    --output_dir results/direct_prediction