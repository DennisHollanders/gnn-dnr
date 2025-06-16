#!/bin/bash
#
# Advanced SLURM Job Manager for Predict-then-Optimize Experiments
# Adapted to use slurm_logs/ and tue.default.q partition

# ----------------------------------------------------------------------------
# CONFIGURATION SECTION - MODIFY THESE PATHS IF NEEDED
# ----------------------------------------------------------------------------

# Base directories
BASE_DIR="$HOME/gnn-dnr"
DATA_DIR="$BASE_DIR/data/split_datasets/test"
LOG_DIR="$BASE_DIR/slurm_logs"

# Make sure the log directory exists
mkdir -p "$LOG_DIR"

# Model configurations
declare -A MODELS=(
  ["GAT"]="$BASE_DIR/model_search/models/final_models/blooming-snow-15-Best.pt"
  ["GIN"]="$BASE_DIR/model_search/models/final_models/cosmic-field-12-Best.pt"
  ["GCN"]="$BASE_DIR/model_search/models/final_models/volcanic-moon-10-Best.pt"
)

declare -A CONFIGS=(
  ["GAT"]="$BASE_DIR/model_search/models/final_models/AdvancedMLP------blooming-snow-15.yaml"
  ["GIN"]="$BASE_DIR/model_search/models/final_models/AdvancedMLP------cosmic-field-12.yaml"
  ["GCN"]="$BASE_DIR/model_search/models/final_models/AdvancedMLP------volcanic-moon-10.yaml"
)

# Experiment parameters
DATASET_NAME="test"
NUM_WORKERS=8
BATCH_SIZE=32

# SLURM parameters
PARTITION="tue.default.q"
TIME_LIMIT="1-00:00:00"
MEMORY="16G"
CPUS=4
NODES=1
NTASKS_PER_NODE=1

SUBMIT_LOG="$LOG_DIR/all_submissions.log"

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------

submit_job() {
    local job_name=$1
    local model_name=$2
    local model_path=$3
    local config_path=$4
    local warmstart_mode=$5
    local rounding_method=$6
    local confidence_threshold=$7
    local predict_flag=$8
    local optimize_flag=$9

    local job_script="$LOG_DIR/${job_name}.slurm"

    cat > "$job_script" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}

# Load modules
module purge
module load Python/3.11.3
module load poetry/1.5.1-GCCcore-12.3.0
module load Gurobi/11.0.3-GCCcore-12.3.0

echo "=== Job Info ==="
echo "Name: ${job_name}"
echo "Model: ${model_name}"
echo "Warmstart: ${warmstart_mode}"
echo "Rounding: ${rounding_method}"
echo "Confidence: ${confidence_threshold}"
echo "Started: \$(date)"
echo "================"

cd "$BASE_DIR" || exit 1
export PYTHONPATH="$BASE_DIR:$BASE_DIR/model_search:\$PYTHONPATH"

poetry run python -I model_search/predict_then_optimize.py \\
  --config_path     "${config_path}" \\
  --model_path      "${model_path}" \\
  --folder_names    "${DATA_DIR}" \\
  --dataset_names   "${DATASET_NAME}" \\
  --batch_size      ${BATCH_SIZE} \\
  --warmstart_mode  "${warmstart_mode}" \\
  --rounding_method "${rounding_method}" \\
  --confidence_threshold ${confidence_threshold} \\
  --num_workers     ${NUM_WORKERS} \\
  ${predict_flag} \\
  ${optimize_flag}

echo "Finished: \$(date)"
EOF

    # Submit it
    if sbatch_out=$(sbatch "$job_script"); then
        local job_id=$(echo "$sbatch_out" | awk '{print $4}')
        echo "$(date '+%F %T') | SUBMITTED | $job_name | $job_id" >> "$SUBMIT_LOG"
        echo "Submitted $job_name (ID $job_id)"
    else
        echo "$(date '+%F %T') | FAILED    | $job_name" >> "$SUBMIT_LOG"
        echo "ERROR submitting $job_name"
    fi
}

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

echo "Starting batch submission: $(date)"
echo "Logging to: $LOG_DIR"
touch "$SUBMIT_LOG"

job_count=0

for model_name in "${!MODELS[@]}"; do
    model_path="${MODELS[$model_name]}"
    config_path="${CONFIGS[$model_name]}"

    # Direct Prediction
    for rounding in "round" "PhyR"; do
        job_name="${model_name}_DirectPred_${rounding}"
        submit_job "$job_name" "$model_name" "$model_path" "$config_path" \
                   "none" "$rounding" "0.5" "--predict" "--optimize"
        ((job_count++))
    done

    # Soft WarmStart (float + binary)
    for mode in "float" "soft"; do
      for rounding in "round" "PhyR"; do
        job_name="${model_name}_SoftWarm_${mode}_${rounding}"
        submit_job "$job_name" "$model_name" "$model_path" "$config_path" \
                   "$mode" "$rounding" "0.5" "--predict" "--optimize"
        ((job_count++))
      done
    done

    # Hard WarmStart (various confidences)
    for confidence in 0.9 0.7 0.5 0.3 0.1; do
      for rounding in "round" "PhyR"; do
        job_name="${model_name}_HardWarm_${confidence}_${rounding}"
        submit_job "$job_name" "$model_name" "$model_path" "$config_path" \
                   "hard" "$rounding" "${confidence}" "--predict" "--optimize"
        ((job_count++))
      done
    done

    echo "=> Submitted $job_count jobs for $model_name"
done

echo "All done. Total jobs submitted: $job_count"
echo "Check status with: squeue -u \$USER or sacct -j <job_id>"
echo "Logs & scripts in: $LOG_DIR"
