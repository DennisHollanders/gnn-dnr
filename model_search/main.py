import argparse
import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from train import train, test
from torch_geometric.data import DataLoader
from load_data import create_data_loaders, DataloaderType
import optuna
import wandb
import importlib
import sys
import logging 
from tqdm import tqdm

console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter("%(message)s"))   # just the text

# ── 2. attach it to your logger ──────────────────────────────────────────────
logger = logging.getLogger("Distribution Network Reconfiguration -- Model Search")
logger.setLevel(logging.INFO)
logger.handlers.clear()          # throw away any old handlers that carry the prefix
logger.addHandler(console)
logger.propagate = False 


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Add the model_search directory to sys.path (might be redundant if ROOT_DIR is added, but safe)
model_search_path = ROOT_DIR / "model_search"
if str(model_search_path) not in sys.path:
    sys.path.append(str(model_search_path))

from evaluation.evaluation import run_evaluation

# Fallback description if none is provided via command line or YAML.
DEFAULT_DESCRIPTION = "Train Graph Model"

def parse_args():
    parser = argparse.ArgumentParser(description=DEFAULT_DESCRIPTION)
    # New argument to allow overriding the description (e.g., "Train PIGNN")
    parser.add_argument("--description", type=str, help="Description for the training job")
    parser.add_argument("--job_name", type=str, default="default_job", help="Job name")
    parser.add_argument("--data_folder", type=str, help="Path to synthetic data folder")
    parser.add_argument("--real_data_folder", type=str, help="Path to real data folder for second validation loop")
    parser.add_argument("--model_module", type=str, default="first_model",                        help="Name of file containing the model class")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32, 16], help="Hidden dimensions")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent dimension")
    parser.add_argument("--activation", type=str, default="prelu",
                        choices=["relu", "leaky_relu", "elu", "selu", "prelu", "sigmoid", "tanh"])
    parser.add_argument("--criterion_name", type=str, default="MSELoss", help="Criterion")
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
    
    parser.add_argument("--loader_type", type=str, default="default", choices=["default", "PINN", "PIGNN", "GNN", "MLP"],)
    parser.add_argument("--batching_type", type=str, default="dynamic", choices=["standard", "dynamic"])
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--max_edges", type=int, default=5000)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()

def save_args_to_yaml(args, model_folder):
    """
    Save the current arguments to a YAML file in a 'config_files' folder 
    inside the model's directory.
    """
    config = vars(args)
    config_dir = os.path.join(model_folder, "config_files")
    os.makedirs(config_dir, exist_ok=True)
    filename = os.path.join(config_dir, f"{args.model_module}------{args.job_name}.yaml")
    with open(filename, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Configuration saved to {filename}")

def load_args_from_yaml(filepath):
    """Load arguments from a YAML config file and return an argparse.Namespace object."""
    filename = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_search"))) / f"config_files" / f"{filepath}"
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)

def main():
    args = parse_args()

    print("args before loading from yaml:", args)
    
    # If a YAML config is provided, load it to override command-line arguments.
    if args.config:
        logger.info(f"Loaded configuration from {args.config}")
        args = load_args_from_yaml(args.config)
        
    print("args after loading from yaml:", args)
    # Import the model module to get the folder in which it resides.
    model_module = importlib.import_module(f"models.{args.model_module}.{args.model_module}")
    model_folder = os.path.dirname(model_module.__file__)
    print("DEBUG: Value of args.model_module before import:", args.model_module)
    try:
        # Assuming model modules are in model_search/models/your_model_name/your_model_name.py
        # The module path should be models.your_model_name.your_model_name
        model_module_path = f"models.{args.model_module}.{args.model_module}"
        print("DEBUG: Constructed module path for import:", model_module_path)

        # Attempt to import the module
        model_module = importlib.import_module(model_module_path)

        # Get the directory where the model module file is located
        model_folder = os.path.dirname(model_module.__file__)

        # Get the model class from the imported module
        model_class = getattr(model_module, args.model_module)
        print(f"Successfully imported model: {args.model_module}")

    except ImportError as e:
        print(f"Error importing model module '{model_module_path}': {e}")
        print("Please ensure your model is in the correct directory structure (e.g., model_search/models/YourModelName/YourModelName.py)")
        print(f"Also check that the module name '{args.model_module}' matches the directory and file name.")
        sys.exit(1) # Exit if model cannot be imported
    except AttributeError as e:
         print(f"Error finding model class '{args.model_module}' in module '{model_module_path}': {e}")
         print("Please ensure the model class name within the file matches the module name.")
         sys.exit(1) # Exit if model class is not found
    except Exception as e:
         print(f"An unexpected error occurred during model import or class retrieval: {e}")
         sys.exit(1)
    # Use provided description if available; otherwise, use the default.
    project_description = args.description if args.description else DEFAULT_DESCRIPTION

    if args.wandb:
        run = wandb.init(project=project_description, job_type="train", config=vars(args))
        args.job_name = run.name    

    # Save the configuration file in the model folder's config_files subdirectory.
    save_args_to_yaml(args, model_folder)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(args.loader_type)
    
    logger.info(args.data_folder)
    logger.info(args.real_data_folder)
    train_loader, val_loader, val_real_loader, test_loader = create_data_loaders(
        base_directory=args.data_folder,
        secondary_directory=args.real_data_folder,
        loader_type=DataloaderType[args.loader_type.upper()], 
        batch_size=args.batch_size,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        batching_type=args.batching_type
    )
    loaders = [train_loader, val_loader, val_real_loader, test_loader]

    model_class = getattr(model_module, args.model_module)

    logger.info("\n +++++++++++++++++++")
    for idx, loader in enumerate(loaders):
        logger.info(f"\n ---printing {loader}  stats:")
        batch = next(iter(loader))
        logger.info(f"keys: {batch.keys()}")
        logger.info(f"batch size: {batch.batch_size}")
        logger.info(f"edge index: {batch.edge_index.shape}")
        logger.info(f"edge attr: {batch.edge_attr.shape}")
        logger.info(f"node attr: {batch.x.shape}")

    node_input_dim = train_loader.dataset[0].x.shape[1]
    edge_input_dim = train_loader.dataset[0].edge_attr.shape[1]

    try:
        criterion_class = getattr(model_module, args.criterion_name)
        try: 
            criterion = criterion_class(weight_switch=1.0, weight_physics=10.0)  
        except:
            criterion = criterion_class()
    except (ImportError, AttributeError) as e:
        logger.debug(f"Error loading {args.criterion_name} from {model_module}: {e}")
        if args.criterion_name == "MSELoss":
            criterion = nn.MSELoss()
        elif "L1Loss":
            criterion = nn.L1Loss()
        elif "SmoothL1Loss":
            criterion = nn.SmoothL1Loss()
    
    model = model_class(
        node_input_dim=node_input_dim,
        edge_input_dim= edge_input_dim, #args.edge_input_dim,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        activation=args.activation,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    best_loss = float("inf")
    patience = 0

    logger.info(f"Training {args.model_module} with {len(train_loader.dataset)} samples")
    logger.info(f"Validation with {len(val_loader.dataset)} samples")
    try:
        logger.info(f"Test with {len(test_loader.dataset)} samples")
        logger.info(f"real_val with {len(val_real_loader.dataset)} samples")
    except AttributeError:
        logger.info("No test dataset provided")

    for epoch in tqdm(range(args.epochs), desc="Training Progress"):
        train_loss, train_dict = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_dict = test(model, val_loader, criterion, device)

        if epoch % 5 == 0:
            if val_real_loader:
                val_real_loss, val_real_dict = test(model, val_real_loader, criterion, device)
                logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Real Loss: {val_real_loss:.4f}")
                if args.wandb:
                    wandb.log({"val_real_loss": val_real_loss, **val_real_dict})
            else:
                logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            if args.wandb:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, **train_dict, **val_dict})
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), f"model_search/models/{args.model_module}/{args.job_name}-Best.pt")
            else:
                patience += 1
                if patience == args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
    # Check if test_loader is a valid DataLoader object before using it
    if test_loader:
        test_loss, test_dict = test(model, test_loader, criterion, device) # Assuming test also returns a dict
        logger.info(f"Test Loss: {test_loss:.4f}")
        if args.wandb:
             wandb.log({"test_loss": test_loss, **test_dict}) # Log test loss and metrics

    run_evaluation(model, train_loader, val_loader, val_real_loader, test_loader, device, args)

    torch.save(model.state_dict(), f"model_search/models/{args.model_module}/{args.job_name}-Epoch{epoch +1}-Last.pt")

if __name__ == "__main__":
   main()
