#!/bin/bash

# Submit only the missing baseline optimization job

BASE_DIR="$HOME/gnn-dnr"
DATA_DIR="$BASE_DIR/data/split_datasets/test"
LOG_DIR="$BASE_DIR/slurm_logs"
mkdir -p "$LOG_DIR"

# Use the same model/config as the global warmstart job (first model)
GLOBAL_MODEL="GAT"
GLOBAL_MP="$BASE_DIR/model_search/models/final_models/whole-paper-13-Best.pt"
GLOBAL_CP="$BASE_DIR/model_search/models/final_models/AdvancedMLP------whole-paper-13.yaml"

# Experiment parameters
DATASET_NAME="test"
NUM_WORKERS=8

# SLURM parameters (same as your main script)
PARTITION="tue.default.q"
TIME_LIMIT="0-20:00:00"
MEMORY="8G"
CPUS=8
NODES=1
NTASKS_PER_NODE=1

JOB_NAME="optimization_without_warmstart"

cat > "$LOG_DIR/${JOB_NAME}.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}

module purge
module load Python/3.11.3
module load poetry/1.5.1-GCCcore-12.3.0
module load Gurobi/11.0.3-GCCcore-12.3.0

echo "=== Job Info ==="
echo "Name: ${JOB_NAME}"
echo "Model: ${GLOBAL_MODEL}"
echo "Warmstart: optimization_without_warmstart"
echo "Rounding: round"
echo "Confidence: 0.5"
echo "Started: \$(date)"
echo "================"

cd "$BASE_DIR" || exit 1
export PYTHONPATH="$BASE_DIR:$BASE_DIR/model_search:\$PYTHONPATH"

poetry run python -I model_search/predict_then_optimize.py \\
  --config_path     "${GLOBAL_CP}" \\
  --model_path      "${GLOBAL_MP}" \\
  --folder_names    "${DATA_DIR}" \\
  --dataset_names   "${DATASET_NAME}" \\
  --warmstart_mode  "optimization_without_warmstart" \\
  --rounding_method "round" \\
  --confidence_threshold 0.5 \\
  --num_workers     ${NUM_WORKERS} \\
  --optimize

echo "Finished: \$(date)"
EOF

sbatch "$LOG_DIR/${JOB_NAME}.slurm"
echo "Submitted baseline job: $JOB_NAME"