#!/bin/bash
#SBATCH --job-name=gnn-train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
# SBATCH --gres=gpu:1 
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --partition=tue.default.q # Or tue.gpu.q if using the --gres=gpu:1 directive

export DATA_FOLDER="test_data"

# Create required directories
mkdir -p ./data
mkdir -p ./logs

echo "Loading modules..."

!
module purge 
module load Python/3.11.3
module load poetry/1.5.1-GCCcore-12.3.0
module load Gurobi/11.0.3-GCCcore-12.3.0
echo "Modules loaded."


echo "Navigating to project directory..."
cd $HOME/gnn-dnr || exit 1 

echo "Running script using 'poetry run'..."

poetry run python -I data_generation/define_ground_truth.py --folder_path "/vast.mnt/home/20174047/gnn-dnr/data/test_val_real__range-30-230_nTest-1000_nVal-1000_2552025_1/test/"

echo "Script finished."

# Deactivate if using Option 1 - not needed for Option 2
# deactivate