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
mkdir -p "$LOG_DIR"

# Model configurations
declare -A MODELS=(
 # ["GAT"]="$BASE_DIR/model_search/models/AdvancedMLP/GAT-stage2-hyperparameter-tuning-Best.pt"
 # ["GIN"]="$BASE_DIR/model_search/models/AdvancedMLP/GIN-stage2-hyperparameter-tuning-Best.pt"
  ["GCN"]="$BASE_DIR/model_search/models/AdvancedMLP/GCN-stage2-hyperparameter-tuning-Best.pt"
)
declare -A CONFIGS=(
 # ["GAT"]="$BASE_DIR/model_search/models/AdvancedMLP/config_files/AdvancedMLP------GAT-stage2-hyperparameter-tuning.yaml"
 # ["GIN"]="$BASE_DIR/model_search/models/AdvancedMLP/config_files/AdvancedMLP------GIN-stage2-hyperparameter-tuning.yaml"
  ["GCN"]="$BASE_DIR/model_search/models/AdvancedMLP/config_files/AdvancedMLP------GCN-stage2-hyperparameter-tuning.yaml"
)

declare -A CPUS_PER_MODEL=(
 # ["GAT"]=32
 # ["GIN"]=32
  ["GCN"]=32
)
declare -A PARTITIONS_PER_MODEL=(
 # ["GAT"]="tue.default.q"
 # ["GIN"]="be.student.q"
  ["GCN"]="elec-ees-empso.cpu.q"
)

# Grab one “first” model for the global warm-start job
keys=("${!MODELS[@]}")
GLOBAL_MODEL="${keys[0]}"
GLOBAL_MP="${MODELS[$GLOBAL_MODEL]}"
GLOBAL_CP="${CONFIGS[$GLOBAL_MODEL]}"

# Experiment parameters
DATASET_NAME="test"
NUM_WORKERS=4

TIME_LIMIT="2-00:00:00"
MEMORY="8G"
NODES=1
NTASKS_PER_NODE=1

SUBMIT_LOG="$LOG_DIR/all_submissions.log"

# ----------------------------------------------------------------------------
# HELPER FUNCTION
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

    local cpus="${CPUS_PER_MODEL[$model_name]}"
    local partition="${PARTITIONS_PER_MODEL[$model_name]}"
    local job_script="$LOG_DIR/${job_name}.slurm"

    cat > "$job_script" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${partition}

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
  --warmstart_mode  "${warmstart_mode}" \\
  --rounding_method "${rounding_method}" \\
  --confidence_threshold ${confidence_threshold} \\
  --num_workers     ${NUM_WORKERS} \\
  ${predict_flag} \\
  ${optimize_flag}

echo "Finished: \$(date)"
EOF

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

choices=("only_gnn_predictions" "soft" "float" "hard")
job_count=0

for model in "${!MODELS[@]}"; do
  mp="${MODELS[$model]}"
  cp="${CONFIGS[$model]}"

  for choice in "${choices[@]}"; do
    case $choice in
      only_gnn_predictions)
        predict="--predict"; optimize=""
        warm="only_gnn_predictions"; conf=0.5
        for round in "round" "PhyR"; do
          name="${model}_only_gnn_${round}"
          submit_job "$name" "$model" "$mp" "$cp" "$warm" "$round" "$conf" "$predict" "$optimize"
          ((job_count++))
        done
        continue
        ;;

      soft)
        predict="--predict"; optimize="--optimize"
        warm="soft"; conf=0.5
        for round in "round" "PhyR"; do
          name="${model}_soft_${round}"
          submit_job "$name" "$model" "$mp" "$cp" "$warm" "$round" "$conf" "$predict" "$optimize"
          ((job_count++))
        done
        continue
        ;;

      float)
        predict="--predict"; optimize="--optimize"
        warm="float"; round="round"; conf=0.5
        ;;

      hard)
        for conf in 0.99 0.975 0.95 0.9; do
          for round in "round" "PhyR"; do
            name="${model}_Hard_${conf}_${round}"
            submit_job "$name" "$model" "$mp" "$cp" "hard" "$round" "$conf" "--predict" "--optimize"
            ((job_count++))
          done
        done
        continue
        ;;
    esac

    # default submit for float
    name="${model}_${choice}"
    submit_job "$name" "$model" "$mp" "$cp" "$warm" "$round" "$conf" "$predict" "$optimize"
    ((job_count++))
  done

  echo "=> Submitted $job_count jobs so far"
done



echo "All done. Total jobs: $job_count"
echo "Check with: squeue -u \$USER or sacct -j <job_id>"
echo "Scripts & logs: $LOG_DIR"
