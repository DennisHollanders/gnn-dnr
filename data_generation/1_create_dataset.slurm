#!/bin/bash
#SBATCH --job-name=gnn-train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
# SBATCH --gres=gpu:1 
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --partition=tue.default.q # Or tue.gpu.q if using the --gres=gpu:1 directive

# Create required directories
mkdir -p ./data
mkdir -p ./logs

echo "Loading modules..."

!
module purge # Start with a clean environment
module load Python/3.11.3
# module load CUDA/11.7.0 
module load poetry/1.5.1-GCCcore-12.3.0
module load Gurobi/11.0.3-GCCcore-12.3.0
echo "Modules loaded."


echo "Navigating to project directory..."
cd $HOME/gnn-dnr || exit 1 # Add error check for cd

echo "Running script using 'poetry run'..."

# run over all subgraphs
poetry run python -I data_generation/create_dataset.py --iterate_all --n_samples_per_graph 5 --logging false
# run over all and generate test val with specific range
#poetry run python -I data_generation/create_dataset.py --iterate_all --n_samples_per_graph 5 --logging false --generate_test_val true --busrange_test_val 80 150

# run over selected subgraphs
#poetry run python -I data_generation/create_dataset.py --num subgraphs 5000 --target_busses 100 -- busrange 10 --logging false

echo "Script finished."

