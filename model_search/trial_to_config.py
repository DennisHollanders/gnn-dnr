import argparse
import ast
import os
import sys
import pandas as pd
import yaml

def convert_value(val):
    # Convert string representations to Python types
    if isinstance(val, str):
        val_strip = val.strip()
        # Try Python literal (lists/dicts)
        if (val_strip.startswith("[") and val_strip.endswith("]")) or \
           (val_strip.startswith("{") and val_strip.endswith("}")):
            try:
                return ast.literal_eval(val_strip)
            except Exception:
                # Fallback: split semicolon-separated lists
                if val_strip.startswith("[") and val_strip.endswith("]") and ";" in val_strip:
                    inner = val_strip[1:-1]
                    items = [p.strip().strip("'\"") for p in inner.split(";") if p.strip()]
                    return items
        # Booleans
        low = val_strip.lower()
        if low in ("true", "false"):
            return low == "true"
        # Numeric
        try:
            if "." in val_strip:
                return float(val_strip)
            return int(val_strip)
        except ValueError:
            pass
        return val
    return val


def generate_config(csv_path: str, trial_num: int, out_path: str):
    """Core logic to generate a single YAML config from a trial in a CSV."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' does not exist.", file=sys.stderr)
        return
    df = pd.read_csv(csv_path, dtype=str)

    # Filter for the requested trial
    df_match = df[df["trial_number"].astype(int) == trial_num]
    if df_match.empty:
        print(f"Error: No entry for trial {trial_num} in {csv_path}.", file=sys.stderr)
        return
    row = df_match.iloc[0].to_dict()

    # Fields to exclude from the final config
    metric_fields = {
        "timestamp", "starting_val_loss", "starting_train_loss", "best_mcc",
        "best_train_loss", "best_val_loss", "best_f1_minority", "best_balanced_accuracy",
        "final_train_loss", "final_val_loss", "final_f1_minority", "final_mcc",
        "final_balanced_accuracy", "final_epoch", "converged", "status", "config_valid"
    }

    # Build config, excluding metric fields
    config = {k: convert_value(v) for k, v in row.items() if k not in metric_fields}
    config['description'] = "GNN Models"

    # Group model-specific parameters into model_kwargs
    root_keys = {
        "trial_number", "job_name", "description", "dataset_names", "folder_names",
        "dataset_type", "batching_type", "max_nodes", "max_edges", "train_ratio",
        "seed", "num_workers", "learning_rate", "weight_decay", "batch_size",
        "epochs", "patience", "criterion_name", "wandb", "model_module", "wandb_project"
    }
    model_kwargs = {k: config.pop(k) for k in list(config.keys()) if k not in root_keys}
    config['model_kwargs'] = model_kwargs

    # Write the config to a YAML file
    with open(out_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Configuration for trial {trial_num} written to {out_path}.")


def generate_single_config_from_cli():
    """Mode 1: Generates a single config using command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a YAML config file from a specified trial in a CSV.")
    parser.add_argument("--csv", required=True,
                        help="Path to the input CSV file containing trial data.")
    parser.add_argument("--trial", type=int, required=True,
                        help="Trial number to extract configuration for.")
    parser.add_argument("--output", default=None,
                        help="Path to write the output YAML config. Defaults to ./config_trial_<trial>.yaml")
    args = parser.parse_args()

    # Determine output path, using default if not provided
    out_path = args.output or f"config_trial_{args.trial}.yaml"

    generate_config(args.csv, args.trial, out_path)


def generate_batch_configs():
    """Mode 2: Generates all 15 predefined configuration files."""
    base_dir = "data/hpos"
    os.makedirs(base_dir, exist_ok=True)

    # Define models, their data folders, and trial numbers
    tasks = {
       # "GCN": {"folder": "hpo_GCN-2-CPU_20250628_101047", "trials": [173,212,137,1,162]},
       # "GAT": {"folder": "hpo_GAT-2-CPU_20250628_101047", "trials": [175, 189, 47, 109, 140]},
       "GIN": {"folder": "hpo_GIN-2-CPU_20250628_174111", "trials": [200]}
    }

    print("--- Running in Batch Mode ---")
    # Process each task
    for model_name, config in tasks.items():
        csv_path = os.path.join(base_dir, config["folder"], "hpo_results.csv")
        for trial in config["trials"]:
            out_file = f"config_{model_name}_trial_{trial}.yaml"
            out_path = os.path.join(base_dir, out_file)
            generate_config(csv_path, trial, out_path)
    print("--- Batch processing complete ---")


if __name__ == "__main__":
    generate_batch_configs()