#!/bin/bash
#SBATCH --job-name=process_subgraphs
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# SBATCH --gres=gpu:1 
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --partition=tue.default.q # Or tue.gpu.q if using the --gres=gpu:1 directive

module purge # Start with a clean environment
module load Python/3.11.3
# module load CUDA/11.7.0 
module load poetry/1.5.1-GCCcore-12.3.0
module load Gurobi/11.0.3-GCCcore-12.3.0
echo "Modules loaded."


# Navigate to your project directory
cd $HOME/gnn-dnr || exit 1 

# Run the processing script
poetry run python -I data_generation/create_buurts.py

echo "Script finished."

