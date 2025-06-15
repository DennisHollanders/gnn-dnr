#!/bin/bash
# submit_all_experiments.sh

echo "Submitting all GIN experiments..."

# Navigate to project directory
cd $HOME/gnn-dnr || exit 1

# Create log and results directories
mkdir -p logs
mkdir -p results

# Check if predictions exist, if not generate them first
if [ ! -f "results/predictions.pkl" ]; then
    echo "Generating predictions first..."
    JOB0=$(sbatch --parsable generate_predictions.sh)
    echo "Submitted prediction generation job: $JOB0"
    DEPENDENCY="--dependency=afterok:$JOB0"
else
    echo "Predictions already exist, skipping generation..."
    DEPENDENCY=""
fi

# Submit direct prediction (fast, can run first)
JOB1=$(sbatch $DEPENDENCY --parsable slurm_direct_prediction.sh)
echo "Submitted direct prediction jobs: $JOB1"

# Submit soft warmstart
JOB2=$(sbatch $DEPENDENCY --parsable slurm_soft_warmstart.sh)
echo "Submitted soft warmstart jobs: $JOB2"

# Submit combined hard warmstart (all thresholds)
JOB3=$(sbatch $DEPENDENCY --parsable slurm_hard_warmstart_all.sh)
echo "Submitted hard warmstart jobs (all thresholds): $JOB3"

echo "All jobs submitted!"
echo "You can monitor progress with: squeue -u $USER"
echo ""
echo "Expected output files:"
echo "  - results/direct_prediction/"
echo "  - results/soft_warmstart/"
echo "  - results/hard_warmstart/"