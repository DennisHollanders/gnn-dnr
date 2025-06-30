#!/usr/bin/env bash
# model_search/hpo.slurm — submit GIN/GAT/GCN HPO with empty mst/mst_opt + .pt caches

set -euo pipefail

# 1) Project root & logs
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/slurm_logs"
mkdir -p "$LOGDIR"

# 2) Helper to submit one study
submit_hpo() {
  CONFIG="$1"   # e.g. hpo-GIN.yaml
  STUDY="$2"    # e.g. GIN
  PART="$3"     # e.g. elec-ees-empso.cpu.q

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
  --trial_timeout 3600
EOF
}

# 3) Submit all three
submit_hpo hpo-GIN.yaml  GIN  elec-ees-empso.cpu.q
submit_hpo hpo-GAT.yaml  GAT  tue.default.q
submit_hpo hpo-GCN.yaml  GCN  elec-ees-empso.cpu.q
echo "✅ All three HPO jobs submitted."
