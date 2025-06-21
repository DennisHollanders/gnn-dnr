#!/bin/bash

# Simple HPO job submission script

BASE_DIR="$HOME/gnn-dnr"
LOG_DIR="$BASE_DIR/slurm_logs"
mkdir -p "$LOG_DIR"

submit_hpo_job() {
    local MODEL=$1
    local CONFIG=$2
    local JOB_NAME="HPO-${MODEL}"
    
    cat > "$LOG_DIR/${JOB_NAME}.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=$LOG_DIR/${JOB_NAME}_%j.out
#SBATCH --error=$LOG_DIR/${JOB_NAME}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --partition=tue.default.q

module purge
module load Python/3.11.3
module load poetry/1.5.1-GCCcore-12.3.0
module load Gurobi/11.0.3-GCCcore-12.3.0

cd $BASE_DIR || exit 1
mkdir -p ./data ./logs

poetry run python -I model_search/hpo.py --config ${CONFIG} --trials 250 --wandb --study_name ${MODEL}
EOF

    sbatch "$LOG_DIR/${JOB_NAME}.slurm"
}

# Submit all three jobs
submit_hpo_job "GAT" "hpo-GAT-stage2.yaml"
submit_hpo_job "GCN" "hpo-GCN-stage2.yaml"
submit_hpo_job "GIN" "hpo-GIN-stage2.yaml"

echo "Submitted 3 HPO jobs: GAT, GCN, GIN"