# load .env
#!/bin/bash
#SBATCH --job-name=gnn_wandb      # Job name
#SBATCH --output=gnn_wandb_%j.log # Output log
#SBATCH --error=gnn_wandb_%j.err  # Error log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu

# Load system modules (CUDA, Python, etc.)
module purge
module load cuda/11.7

# Activate Poetry environment (no manual dependency handling needed)
export PYTHONPATH=/path/to/your/project:$PYTHONPATH
source $(poetry env info --path)/bin/activate

# Set WandB API key 
export WANDB_API_KEY = 2c88dc1e7e7f2e8531ddf5de32c2aeddb391deee

# Run with WandB integration
python gnn_train_wandb.py