#!/bin/bash

# CPU-optimized HPO job submission script

BASE_DIR="$HOME/gnn-dnr"
LOG_DIR="$BASE_DIR/slurm_logs"
mkdir -p "$LOG_DIR"

submit_hpo_job() {
    local MODEL=$1
    local CONFIG=$2
    local JOB_NAME="HPO-${MODEL}-16"
    
    cat > "$LOG_DIR/${JOB_NAME}.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=$LOG_DIR/${JOB_NAME}_%j.out
#SBATCH --error=$LOG_DIR/${JOB_NAME}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=1-00:00:00
#SBATCH --partition=elec-ees-empso.cpu.q 

# Load modules
module purge
module load Python/3.11.3
module load poetry/1.5.1-GCCcore-12.3.0
module load Gurobi/11.0.3-GCCcore-12.3.0

# Set environment variables for CPU optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Force PyTorch to use CPU
export CUDA_VISIBLE_DEVICES=""

# Change to project directory
cd $BASE_DIR || exit 1
mkdir -p ./data ./logs

# Run HPO with optimized parallel settings for CPU
# Using 16 parallel workers (half of 32 CPUs to avoid oversubscription)
poetry run python -I model_search/hpo.py \\
    --config ${CONFIG} \\
    --trials 1 \\
    --study_name ${MODEL}-CPU
     --n_parallel 1 \\
    --trial_timeout 54000

echo "Job completed at: \$(date)"
EOF

    sbatch "$LOG_DIR/${JOB_NAME}.slurm"
}

# Submit all three jobs
submit_hpo_job "GAT-solo" "hpo-GAT.yaml"
submit_hpo_job "GCN-solo" "hpo-GCN.yaml"
submit_hpo_job "GIN-solo" "hpo-GIN.yaml"

echo "Submitted 3 CPU-optimized HPO jobs: GAT, GCN, GIN"