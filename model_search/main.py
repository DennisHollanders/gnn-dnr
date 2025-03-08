import argparse
import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from train import  train, test
from load_data import get_pyg_loader, split_dataset
from torch_geometric.data import DataLoader
import optuna
import wandb
import importlib


DESCRIPTION = "Train Graph Autoencoder" 

def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--job_name", type=str, default="default_job", help="Job name")
    parser.add_argument("--data_folder", type=str, help="Path to synthetic data folder")
    parser.add_argument("--real_data_folder", type=str, help="Path to real data folder for second validation loop")
    parser.add_argument("--model_module", type=str, default="first_model",
                        help="Name of file containing the model class")
    parser.add_argument("--input_dim", type=int, help="Input dimension")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32, 16], help="Hidden dimensions")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent dimension")
    parser.add_argument("--activation", type=str, default="prelu",
                        choices=["relu", "leaky_relu", "elu", "selu", "prelu", "sigmoid", "tanh"])
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--hp_search", action="store_true", help="Enable hyperparameter search with Optuna")
    parser.add_argument("--hp_search_n_trials", type=int, default=50, help="Number of trials for hyperparameter search")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--config", type=str, help="Path to YAML config file (overrides arguments)")
    return parser.parse_args()

def save_args_to_yaml(args):
    """Save the current arguments to a YAML file named with the wandb job name."""
    config = vars(args)
    filename = os.path.join("model_search","config_files", f"{args.model_module}------{args.job_name}.yaml")
    with open(filename, "w") as f:
        yaml.dump(config, f)
    print(f"Configuration saved to {filename}")

def load_args_from_yaml(filepath):
    """Load arguments from a YAML config file and return an argparse.Namespace object."""
    filename = os.path.join("model_search","config_files", filepath)
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)

def main():
    args = parse_args()
    
    # If a YAML config is provided, load it to override command-line arguments.
    if args.config:
        args = load_args_from_yaml(args.config)

    run = wandb.init(project=DESCRIPTION, job_type="train",config=vars(args))
    args.job_name = run.name    

    save_args_to_yaml(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = get_pyg_loader(args.data_folder, batch_size=args.batch_size, shuffle=True).dataset
    train_set, val_set, test_set = split_dataset(full_dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model_module = importlib.import_module(f"models.{args.model_module}.{args.model_module}")
    model_class = getattr(model_module, args.model_module)

    model = model_class(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        activation=args.activation,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    best_loss = float("inf")
    patience = 0

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = test(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), f"model_search/models/{args.model_module}/{run.name}-Best.pt")
        else:
            patience += 1
            if patience == args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
    test_loss = test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    torch.save(model.state_dict(),f"model_search/models/{args.model_module}/{run.name}-Epoch{epoch +1}-Last.pt")

if __name__ == "__main__":
    main()
