#!/bin/bash

# Advanced SLURM Job Manager for Predict-then-Optimize Experiments
# This script creates separate jobs for each experiment combination

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PATHS
# ============================================================================

# Base directories
BASE_DIR="\$HOME/gnn-dnr"
DATA_DIR="\$HOME/gnn-dnr/data/source_datasets/test_val_real__range-30-150_nTest-10_nVal-10_2732025_32/test"
LOG_DIR="\$HOME/gnn-dnr/slurm_logs"

# Model configurations
declare -A MODELS
MODELS["GAT"]="/vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/blooming-snow-15-Best.pt"
MODELS["GIN"]="/vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/cosmic-field-12-Best.pt"
MODELS["GCN"]="/vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/volcanic-moon-10-Best.pt"

declare -A CONFIGS
CONFIGS["GAT"]="/vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/AdvancedMLP------blooming-snow-15.yaml"
CONFIGS["GIN"]="/vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/AdvancedMLP------cosmic-field-12.yaml"
CONFIGS["GCN"]="/vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/AdvancedMLP------volcanic-moon-10.yaml"

# Experiment parameters
DATASET_NAME="test"
NUM_WORKERS=8

# SLURM parameters
PARTITION="tue.default.q"
TIME_LIMIT="1-00:00:00"
MEMORY="16G"
CPUS=16
NODES=1
NTASKS_PER_NODE=1

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

create_log_dir() {
    mkdir -p "$LOG_DIR"
    echo "Created log directory: $LOG_DIR"
}

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
    
    # Create individual job script
    cat > "$job_script" << EOF
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

# Load necessary modules
echo "Loading modules..."
module purge # Start with a clean environment
module load Python/3.11.3
module load poetry/1.5.1-GCCcore-12.3.0
module load Gurobi/11.0.3-GCCcore-12.3.0
echo "Modules loaded."

echo "=== Job Information ==="
echo "Job Name: ${job_name}"
echo "Model: ${model_name}"
echo "Model Path: ${model_path}"
echo "Config Path: ${config_path}"
echo "Warmstart Mode: ${warmstart_mode}"
echo "Rounding Method: ${rounding_method}"
echo "Confidence Threshold: ${confidence_threshold}"
echo "Predict: ${predict_flag}"
echo "Optimize: ${optimize_flag}"
echo "Started at: \$(date)"
echo "======================="

echo "Navigating to project directory..."
cd \$HOME/gnn-dnr || exit 1 # Add error check for cd

echo "Running script using 'poetry run'..."

poetry run python -I predict_then_optimize.py \\
    --config_path "${config_path}" \\
    --model_path "${model_path}" \\
    --folder_names "${DATA_DIR}" \\
    --dataset_names "${DATASET_NAME}" \\
    --batch_size ${BATCH_SIZE} \\
    --warmstart_mode "${warmstart_mode}" \\
    --rounding_method "${rounding_method}" \\
    --confidence_threshold ${confidence_threshold} \\
    --num_workers ${NUM_WORKERS} \\
    ${predict_flag} \\
    ${optimize_flag}

echo "Completed at: \$(date)"
EOF

    # Submit the job
    local job_id=$(sbatch "$job_script" | awk '{print $4}')
    echo "Submitted job: $job_name (ID: $job_id)"
    return 0
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    echo "Starting SLURM job submission for predict-optimize experiments"
    echo "============================================================="
    
    create_log_dir
    
    local job_count=0
    local submitted_jobs=()
    
    # Iterate through all model combinations
    for model_name in "${!MODELS[@]}"; do
        model_path="${MODELS[$model_name]}"
        config_path="${CONFIGS[$model_name]}"
        
        echo "Processing model: $model_name"
        
        # 1. DirectPrediction experiments (none warmstart)
        for rounding in "round" "PhyR"; do
            job_name="${model_name}_DirectPred_${rounding}"
            submit_job "$job_name" "$model_name" "$model_path" "$config_path" \
                      "none" "$rounding" "0.5" "--predict" "--optimize"
            submitted_jobs+=("$job_name")
            ((job_count++))
        done
        
        # 2. SoftWarmStart experiments
        # Float warmstart
        for rounding in "round" "PhyR"; do
            job_name="${model_name}_SoftWarm_Float_${rounding}"
            submit_job "$job_name" "$model_name" "$model_path" "$config_path" \
                      "float" "$rounding" "0.5" "--predict" "--optimize"
            submitted_jobs+=("$job_name")
            ((job_count++))
        done
        
        # Binary soft warmstart
        for rounding in "round" "PhyR"; do
            job_name="${model_name}_SoftWarm_Binary_${rounding}"
            submit_job "$job_name" "$model_name" "$model_path" "$config_path" \
                      "soft" "$rounding" "0.5" "--predict" "--optimize"
            submitted_jobs+=("$job_name")
            ((job_count++))
        done
        
        # 3. HardWarmStart experiments with different confidence thresholds
        for confidence in "0.9" "0.7" "0.5" "0.3" "0.1"; do
            for rounding in "round" "PhyR"; do
                job_name="${model_name}_HardWarm_${confidence}_${rounding}"
                submit_job "$job_name" "$model_name" "$model_path" "$config_path" \
                          "hard" "$rounding" "$confidence" "--predict" "--optimize"
                submitted_jobs+=("$job_name")
                ((job_count++))
            done
        done
        
        echo "Submitted $((job_count)) jobs for $model_name"
        echo "---"
    done
    
    echo "============================================================="
    echo "Total jobs submitted: $job_count"
    echo "Job scripts created in: $LOG_DIR"
    echo ""
    echo "Submitted jobs:"
    printf '%s\n' "${submitted_jobs[@]}"
    echo ""
    echo "Monitor jobs with: squeue -u \$USER"
    echo "Check job status: sacct -j <job_id>"
    echo "============================================================="
}

# ============================================================================
# EXECUTION WITH COMMAND LINE OPTIONS
# ============================================================================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            echo "DRY RUN MODE - No jobs will be submitted"
            DRY_RUN=true
            shift
            ;;
        --models)
            IFS=',' read -ra SELECTED_MODELS <<< "$2"
            echo "Selected models: ${SELECTED_MODELS[*]}"
            shift 2
            ;;
        --warmstart-modes)
            IFS=',' read -ra SELECTED_WARMSTART <<< "$2"
            echo "Selected warmstart modes: ${SELECTED_WARMSTART[*]}"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dry-run                    Show what would be submitted without submitting"
            echo "  --models GAT,GIN,GCN         Select specific models (default: all)"
            echo "  --warmstart-modes none,soft  Select specific warmstart modes"
            echo "  --help                       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Override submit_job function for dry run
if [ "$DRY_RUN" = true ]; then
    submit_job() {
        local job_name=$1
        echo "[DRY RUN] Would submit job: $job_name"
        echo "  Model: $2, Warmstart: $5, Rounding: $6, Confidence: $7"
        return 0
    }
fi

# Run main function
main

echo "Script completed at: $(date)"