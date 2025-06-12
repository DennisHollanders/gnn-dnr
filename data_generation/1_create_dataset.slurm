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
# poetry run python -I data_generation/create_dataset.py --generate_synthetic_data true --num_subgraphs 2500 --target_busses 130  --bus_range 100 --logging True
# run over all and generate test val with specific range

#poetry run python -I data_generation/create_dataset.py --generate_synthetic_data true iterate_all --n_samples_per_graph 5 --logging True

poetry run python -I data_generation/create_dataset.py --sample_real_data true --logging true  --bus_range_test_val 30 230 --samples_per_dataset [1000,1000]
# --dataset_names ["test","validation","train"] --samples_per_dataset [1000,1000,10000]

echo "Script finished."

