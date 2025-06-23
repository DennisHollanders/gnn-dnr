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


def main():
    parser = argparse.ArgumentParser(
        description="Generate a YAML config file from a specified trial in a CSV.")
    parser.add_argument("--csv", required=True,
                        help="Path to the input CSV file containing trial data.")
    parser.add_argument("--trial", type=int, required=True,
                        help="Trial number to extract configuration for.")
    parser.add_argument("--output", default=None,
                        help="Path to write the output YAML config. Defaults to ./config_trial_<trial>.yaml")
    args = parser.parse_args()

    # Load CSV
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' does not exist.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(args.csv, dtype=str)

    # Filter for the requested trial
    df_match = df[df["trial_number"].astype(int) == args.trial]
    if df_match.empty:
        print(f"Error: No entry found for trial number {args.trial}.", file=sys.stderr)
        sys.exit(1)
    if len(df_match) > 1:
        print(f"Warning: Multiple entries found for trial {args.trial}. Using the first.", file=sys.stderr)
    row = df_match.iloc[0].to_dict()

    # Define metric fields to exclude from config
    metric_fields = {
        "timestamp", "starting_val_loss", "starting_train_loss", "best_mcc",
        "best_train_loss", "best_val_loss", "best_f1_minority", "best_balanced_accuracy",
        "final_train_loss", "final_val_loss", "final_f1_minority", "final_mcc",
        "final_balanced_accuracy", "final_epoch", "converged", "status", "config_valid"
    }

    # Build flat config dict
    config = {}
    for key, value in row.items():
        if key in metric_fields:
            continue
        config[key] = convert_value(value)

    # Override description
    config['description'] = "GNN Models"

    # Group model-specific parameters under model_kwargs
    root_keys = {
        "trial_number", "job_name", "description", "dataset_names", "folder_names",
        "dataset_type", "batching_type", "max_nodes", "max_edges", "train_ratio",
        "seed", "num_workers", "learning_rate", "weight_decay", "batch_size",
        "epochs", "patience", "criterion_name", "wandb", "model_module",
        "wandb_project"
    }
    model_kwargs = {}
    for key in list(config.keys()):
        if key not in root_keys:
            model_kwargs[key] = config.pop(key)
    config['model_kwargs'] = model_kwargs

    # Determine output path
    out_path = args.output or f"config_trial_{args.trial}.yaml"

    # Write YAML
    with open(out_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Configuration for trial {args.trial} written to {out_path}.")


if __name__ == "__main__":
    main()
