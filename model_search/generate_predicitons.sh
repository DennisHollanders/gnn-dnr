#!/bin/bash
#SBATCH --job-name=generate_predictions
#SBATCH --output=logs/predictions_%j.out
#SBATCH --error=logs/predictions_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=tue.default.q

echo "Loading modules..."
module purge
module load Python/3.11.3
module load poetry/1.5.1-GCCcore-12.3.0
echo "Modules loaded."

# Navigate to project directory
cd $HOME/gnn-dnr || exit 1

# Create results directory
mkdir -p ./results

echo "Generating predictions..."
poetry run python -c "
from predict_then_optimize import Predictor
import pickle

# Initialize predictor
predictor = Predictor(
    model_name='GIN',
    model_path='/vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/None-Best.pt',
    model_config='/vast.mnt/home/20174047/gnn-dnr/model_search/models/final_models/config-mlp.yaml',
    folder_name='/vast.mnt/home/20174047/gnn-dnr/data/split_datasets/test/',
    output_folder='./results'
)

# Generate predictions
predictions = predictor.generate_predictions()

# Save predictions
with open('./results/predictions.pkl', 'wb') as f:
    pickle.dump(predictions, f)

print(f'Generated predictions for {len(predictions)} networks')
"