#!/usr/bin/env bash

set -euo pipefail

# 1) Project root & logs
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/slurm_logs"
mkdir -p "$LOGDIR"

# 2) Helper to submit one study
submit_hpo() {
  CONFIG="$1"   
  STUDY="$2"    
  PART="$3"    
  CHECKPOINT="$4"

  echo "→ Submitting HPO ${STUDY} → ${PART}"
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=HPO-${STUDY}
#SBATCH --partition=${PART}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=${LOGDIR}/HPO-${STUDY}_%j.out
#SBATCH --error=${LOGDIR}/HPO-${STUDY}_%j.err

module purge
module load Python/3.11.3
module load poetry/1.5.1-GCCcore-12.3.0

# Stage empty mst/mst_opt and .pt caches into node-local /tmp
DATA_LOCAL=/tmp/\${SLURM_JOB_ID}_data

# Make split dirs plus empty mst & mst_opt
mkdir -p "\${DATA_LOCAL}/split_datasets/train/mst" \
         "\${DATA_LOCAL}/split_datasets/train/mst_opt" \
         "\${DATA_LOCAL}/split_datasets/validation/mst" \
         "\${DATA_LOCAL}/split_datasets/validation/mst_opt" \
         "\${DATA_LOCAL}/split_datasets/test/mst" \
         "\${DATA_LOCAL}/split_datasets/test/mst_opt"

# Copy only root-level .pt cache files
for split in train validation test; do
  cp -v "\$HOME/gnn-dnr/data/split_datasets/\$split/"*.pt \
        "\${DATA_LOCAL}/split_datasets/\$split/" 2>/dev/null || true
done

export DATA_DIR="\${DATA_LOCAL}/split_datasets"

# Pin each Optuna trial to one core
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${ROOT}"
poetry run python -I model_search/hpo.py \
  --config ${CONFIG} \
  --study_name ${STUDY}-CPU \
  --n_parallel 32 \
  --trials 200 \
  --trial_timeout 3600 \
  --stage1_checkpoint ${CHECKPOINT}
EOF
}

# 3) Submit all three
submit_hpo hpo-GIN-stage2.yaml  GIN  elec-ees-empso.cpu.q "/vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/GIN_model_trial_188.pt"
submit_hpo hpo-GAT-stage2.yaml  GAT  tue.default.q "/vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/GAT_model_trial_199.pt"
submit_hpo hpo-GCN-stage2.yaml  GCN  elec-ees-empso.cpu.q "/vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/GCN_model_trial_121.pt"
echo "✅ All three HPO jobs submitted."
