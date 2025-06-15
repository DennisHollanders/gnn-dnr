#!/bin/bash
#SBATCH --job-name=debugPredict
#SBATCH --output=slurm_logs/debugPredict_%j.out
#SBATCH --error=slurm_logs/debugPredict_%j.err
#SBATCH --partition=tue.default.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# --- load your modules ---
module purge
module load Python/3.11.3
module load poetry/1.5.1-GCCcore-12.3.0
module load Gurobi/11.0.3-GCCcore-12.3.0

echo "Debug SLURM job started at $(date)"

# --- go to project root ---
cd $HOME/gnn-dnr

# --- ensure Python can find both model_search/* and its sibling scripts ---
export PYTHONPATH="$HOME/gnn-dnr:$HOME/gnn-dnr/model_search:$PYTHONPATH"

# --- run one experiment via module invocation ---
poetry run python -I model_search/predict_then_optimize.py \
  --config_path  "/vast.mnt/home/20174047/gnn-dnr/model_search/config_files/config-GIN.yaml" \
  --model_path   "/vast.mnt/home/20174047/gnn-dnr/model_search/models/GIN/best_model.pt" \
  --folder_names "/vast.mnt/home/20174047/gnn-dnr/data/source_datasets/test_val_real__range-30-150_nTest-10_nVal-10_2732025_32/test" \
  --dataset_names "test" \
  --warmstart_mode       "hard" \
  --rounding_method      "round" \
  --confidence_threshold 0.7 \
  --num_workers          1 \
  --predict \
  --optimize

echo "Debug SLURM job finished at $(date)"
