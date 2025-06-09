import argparse
from ast import arg
import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from train import train, test
import optuna
import wandb
import importlib
import sys
import logging 



# :TODO WEIGHT INITIALIZATION
# :TODO OPTUNA HYPERPARAMETER SEARCH
# :TODO FIX LOGGING TRAIN
# :TODO Implement Single commodity flow to convex model



console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter("%(message)s"))   

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    )
# ── 2. attach it to your logger ──────────────────────────────────────────────
logger = logging.getLogger("Distribution Network Reconfiguration -- Model Search")
logger.setLevel(logging.INFO)
logger.handlers.clear()         
logger.addHandler(console)
logger.propagate = False 


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Add the model_search directory to sys.path (might be redundant if ROOT_DIR is added, but safe)
model_search_path = ROOT_DIR / "model_search"
if str(model_search_path) not in sys.path:
    sys.path.append(str(model_search_path))

from load_data import create_data_loaders
from evaluation.evaluation import run_evaluation
from src.cvxpy_SOCP import build_convex_problem

DEFAULT_DESCRIPTION = "Graph Neural Network Model Search for Distribution Network Reconfiguration"

def parse_args():
    parser = argparse.ArgumentParser(description=DEFAULT_DESCRIPTION)
    # New argument to allow overriding the description (e.g., "Train PIGNN")
    parser.add_argument("--description", type=str, help="Description for the training job")
    parser.add_argument("--job_name", type=str, default="default_job", help="Job name")
    parser.add_argument("--dataset_names", type=str, nargs="+", default= ["train","validation","test"], help="Names of datasets to create loaders for")
    parser.add_argument("--folder_names", type=str, nargs="+", default=[
                "data\\test_val_real__range-30-230_nTest-1000_nVal-1000_2552025_1\\test",
                "data\\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32\\test",
                "data\\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32\\test"], help="Names of folders to look for datasets in")
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
    parser.add_argument("--evaluate_every_x_epoch", type=int, default=5, help="Epoch interval for validation")
    
    parser.add_argument("--dataset_type", type=str, default="default", choices=["default", "PINN", "PIGNN", "GNN", "MLP"],)
    parser.add_argument("--batching_type", type=str, default="dynamic", choices=["standard", "dynamic"])
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--max_edges", type=int, default=5000)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--override_job_name", type=str, help="Override job name for saving the model")
    parser.add_argument("--model_path", type=str, help="Path to the saved model")
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
    return filename

def load_args_from_yaml(filepath):
    """Load arguments from a YAML config file and return an argparse.Namespace object."""
    filename = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_search"))) / f"config_files" / f"{filepath}"
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)

def main():
    args = parse_args()

    print("args before loading from yaml:", args)

    if args.config:
        logger.info(f"Loaded configuration from {args.config}")
        args = load_args_from_yaml(args.config)
        
    print("args after loading from yaml:", args)
    model_module = importlib.import_module(f"models.{args.model_module}.{args.model_module}")
    model_folder = os.path.dirname(model_module.__file__)
    print("DEBUG: Value of args.model_module before import:", args.model_module)

    # Get the model class from the imported module
    model_class = getattr(model_module, args.model_module)
    print(f"Successfully imported model: {args.model_module}")


    project_description = args.description if args.description else DEFAULT_DESCRIPTION

    if args.wandb:
        run = wandb.init(project=project_description, job_type="train", config=vars(args))
        args.job_name = run.name    

    # Save the configuration file in the model folder's config_files subdirectory.
    file_name_yaml = save_args_to_yaml(args, model_folder)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(args.dataset_type)
    logger.info(args.dataset_names)
    logger.info(args.folder_names)

    dataloaders = create_data_loaders(
        dataset_names=args.dataset_names,
        folder_names=args.folder_names,
        dataset_type=args.dataset_type, 
        batch_size=args.batch_size,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        batching_type=args.batching_type,
    )
    train_loader, validation_loader, test_loader = dataloaders.get("train"),  dataloaders.get("validation"), dataloaders.get("test")

    batch = next(iter(train_loader))
    logger.info(f"Batch type: {type(batch)}")
    logger.info(f"Batch: {batch}")
    logger.info(f"Batch keys: {batch.keys() if isinstance(batch, dict) else 'N/A'}")
    logger.info(f"Batch x shape: {batch.x.shape if hasattr(batch, 'x') else 'N/A'}")
    logger.info(f"Batch edge_index shape: {batch.edge_index.shape if hasattr(batch, 'edge_index') else 'N/A'}")
    logger.info(f"Batch edge_attr shape: {batch.edge_attr.shape if hasattr(batch, 'edge_attr') else 'N/A'}")
    logger.info(f"Batch edge_y shape: {batch.edge_y.shape if hasattr(batch, 'edge_y') else 'N/A'}")
    logger.info(f"Batch size: {len(train_loader.dataset)}")

    model_class = getattr(model_module, args.model_module)
    node_input_dim = train_loader.dataset[0].x.shape[1]
    edge_input_dim = train_loader.dataset[0].edge_attr.shape[1]
    args.model_kwargs["node_input_dim"] = node_input_dim
    args.model_kwargs["edge_input_dim"] = edge_input_dim

    model_kwargs = args.model_kwargs
    if args.model_module =="cvx": 
        from models.cvx.cvx import build_cvx_layer
        model_kwargs["cvx_layer"] = build_cvx_layer(next(iter(train_loader)), args)
    
    model = model_class(**model_kwargs).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    best_loss = float("inf")
    patience = 0

    logger.info(f"Training {args.model_module} with {len(train_loader.dataset)} samples")
    logger.info(f"Validation with {len(validation_loader.dataset)} samples")
    try:
        logger.info(f"Test with {len(test_loader.dataset)} samples")
        #logger.info(f"real_val with {len(validation_real_loader.dataset)} samples")
    except AttributeError:
        logger.info("No test dataset provided")

    criterion = nn.MSELoss()

    for epoch in range(args.epochs): # tqdm(range(args.epochs), desc="Training Progress"):
        train_loss, train_dict = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_dict = test(model, validation_loader, criterion, device)

        if epoch % 5 == 0:
            if validation_loader:
                val_real_loss, val_real_dict = test(model, validation_loader, criterion, device)
                logger.info(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Real Loss: {val_real_loss:.4f}")
                if args.wandb:
                    wandb.log({"val_real_loss": val_real_loss, **val_real_dict})
            else:
                logger.info(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
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
        else:
            logger.info(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    # Check if test_loader is a valid DataLoader object before using it
    if test_loader:
        test_loss, test_dict = test(model, test_loader, criterion, device) 
        logger.info(f"Test Loss: {test_loss:.4f}")
        if args.wandb:
             wandb.log({"test_loss": test_loss, **test_dict}) 
    logger.info(f"Training completed. Best validation loss: {best_loss:.4f}")

    model_save_path = f"model_search/models/{args.model_module}/{args.job_name}-Epoch{epoch}-Last.pt"
    torch.save(model.state_dict(),model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    try:
        # add model save_path to args
        args.model_path = model_save_path
        args.config_path = file_name_yaml 
        args.override_job_name = args.model_module + "------" + args.job_name

    
        run_evaluation(model, train_loader, validation_loader, test_loader, device, args)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")   
if __name__ == "__main__":
   main()
