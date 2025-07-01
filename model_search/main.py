import argparse
from ast import Lambda, arg
import os
from numpy import gradient
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import optuna
import wandb
import importlib
import sys
import logging

from torch.utils.tensorboard import SummaryWriter


console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter("%(message)s"))   

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    )
logger = logging.getLogger("Distribution Network Reconfiguration -- Model Search")
logger.setLevel(logging.INFO)
logger.handlers.clear()         
logger.addHandler(console)
logger.propagate = False 


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

model_search_path = ROOT_DIR / "model_search"
if str(model_search_path) not in sys.path:
    sys.path.append(str(model_search_path))

from load_data import create_data_loaders
from evaluation.evaluation import run_evaluation
from train import train, test
from models.cvx.cvx import build_cvx_layer

src_path = ROOT_DIR / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from loss_functions import WeightedBCELoss, FocalLoss

DEFAULT_DESCRIPTION = "Graph Neural Network Model Search for Distribution Network Reconfiguration"

MODEL_KWARGS_KEYS = [
    'output_type', 'num_classes', 'gnn_type', 'gnn_layers', 'gnn_hidden_dim',
    'gat_heads', 'dropout_rate', 'gat_dropout', 'gin_eps', 'node_hidden_dims',
    'use_node_mlp', 'edge_hidden_dims', 'use_edge_mlp', 'activation',
    'use_batch_norm', 'use_residual', 'use_skip_connections', 'pooling',
    'switch_head_type', 'switch_head_layers', 'switch_attention_heads',
    'use_gated_mp', 'use_phyr', 'enforce_radiality', 'phyr_k_ratio',
    'gat_v2', 'gat_edge_dim', 'gin_train_eps', 'gin_mlp_layers',
    'criterion_name'
]

# Define which parameters are architectural vs training hyperparameters
ARCHITECTURE_KEYS = [
    'output_type', 'num_classes', 'gnn_type', 'gnn_layers', 'gnn_hidden_dim',
    'gat_heads', 'gin_eps', 'node_hidden_dims', 'use_node_mlp', 'edge_hidden_dims', 
    'use_edge_mlp', 'activation', 'use_residual', 'use_skip_connections', 'pooling',
    'switch_head_type', 'switch_head_layers', 'switch_attention_heads',
    'use_gated_mp', 'use_phyr', 'enforce_radiality', 'phyr_k_ratio',
    'gat_v2', 'gat_edge_dim', 'gin_train_eps', 'gin_mlp_layers'
]

TRAINING_HYPERPARAMETER_KEYS = [
    'learning_rate', 'weight_decay', 'dropout_rate', 'gat_dropout', 'batch_size',
    'criterion_name', 'lambda_phy_loss', 'lambda_mask', 'lambda_connectivity',
    'lambda_radiality', 'loss_scaling_strategy', 'normalization_type', 'use_batch_norm'
]


def parse_args():
    parser = argparse.ArgumentParser(description=DEFAULT_DESCRIPTION)
    parser.add_argument("--description", type=str, help="Description for the training job")
    parser.add_argument("--job_name", type=str, default="default_job", help="Job name")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--hp_search", action="store_true", help="Enable hyperparameter search with Optuna")
    parser.add_argument("--hp_search_n_trials", type=int, default=50, help="Number of trials for hyperparameter search")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--config", type=str, help="Path to YAML config file (overrides arguments)")
    parser.add_argument("--dataset_type", type=str, default="default", choices=["default", "PINN", "PIGNN", "GNN", "MLP"],)
    parser.add_argument("--batching_type", type=str, default="dynamic", choices=["standard", "dynamic"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--override_job_name", type=str, help="Override job name for saving the model")
    parser.add_argument("--model_path", type=str, help="Path to the saved model")
    parser.add_argument("--stage1_checkpoint", type=str, help="Path to first-stage checkpoint (.pt) file")
    parser.add_argument("--stage2_checkpoint", type=str, help="Path to second-stage checkpoint (.pt) file")
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
    logger.info(f"Loaded CLI args: {vars(args)}")

    # Initialize variables
    model_state_dict = None
    stage1_architecture = {}
    stage2_hyperparams = {}
    
    # 1) Load stage-1 checkpoint (architecture + initial weights)
    if args.stage1_checkpoint:
        logger.info(f"Loading stage1 checkpoint from {args.stage1_checkpoint}")
        ck1 = torch.load(args.stage1_checkpoint, map_location="cpu")
        
        # Extract architecture parameters from stage 1
        stage1_config = ck1.get('config', {})
        for key in ARCHITECTURE_KEYS:
            if key in stage1_config:
                stage1_architecture[key] = stage1_config[key]
        
        # Store the model weights
        model_state_dict = ck1['model_state_dict']
        
        logger.info(f"Loaded architecture from stage 1: {stage1_architecture}")

    # 2) Load stage-2 checkpoint (training hyperparameters)
    if args.stage2_checkpoint:
        logger.info(f"Loading stage2 checkpoint from {args.stage2_checkpoint}")
        ck2 = torch.load(args.stage2_checkpoint, map_location="cpu")
        
        # Extract training hyperparameters from stage 2
        stage2_config = ck2.get('config', {})
        for key in TRAINING_HYPERPARAMETER_KEYS:
            if key in stage2_config:
                stage2_hyperparams[key] = stage2_config[key]
        
        # Also get other training-related configs
        for key in ['epochs', 'patience', 'dataset_names', 'folder_names']:
            if key in stage2_config:
                stage2_hyperparams[key] = stage2_config[key]
        
        logger.info(f"Loaded hyperparameters from stage 2: {stage2_hyperparams}")

    # 3) Build final configuration
    # Start with CLI args
    final_config = vars(args).copy()
    
    # Override with stage 2 hyperparameters
    final_config.update(stage2_hyperparams)
    
    # Override with stage 1 architecture (this ensures we use the correct architecture)
    final_config.update(stage1_architecture)
    #final_config["folder_names"] = ["data/split_datasets/train", "data/split_datasets/validation", "data/split_datasets/test"]
    # If a separate config file is specified, load it last
    if final_config.get('config'):
        logger.info(f"Loading additional config from YAML {final_config['config']}")
        with open(final_config['config'], 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        # Only update keys that aren't already set
        for k, v in yaml_cfg.items():
            if k not in final_config or final_config[k] is None:
                final_config[k] = v

    # Extract model_kwargs from the final config
    model_kwargs = {}
    for key in MODEL_KWARGS_KEYS:
        if key in final_config:
            model_kwargs[key] = final_config[key]
    
    # Update final config with model_kwargs
    final_config['model_kwargs'] = model_kwargs
    final_config['model_module'] = final_config.get('model_module', 'AdvancedMLP')  
    # 5) Reconstruct args namespace
    args = argparse.Namespace(**final_config)
    logger.info(f"Final merged configuration:")
    logger.info(f"  Architecture parameters: {[(k, v) for k, v in model_kwargs.items() if k in ARCHITECTURE_KEYS]}")
    logger.info(f"  Training hyperparameters: {[(k, getattr(args, k, None)) for k in TRAINING_HYPERPARAMETER_KEYS if hasattr(args, k)]}")

    # 6) Build data loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = create_data_loaders(
        dataset_names=args.dataset_names,
        folder_names=args.folder_names,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        max_nodes=20000,
        max_edges=50000,
        train_ratio=0.85,
        seed=0,
        num_workers=0,
        batching_type="standard",
        shuffle=True,
    )
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("validation")
    test_loader = dataloaders.get("test")

    # 7) Instantiate model with stage 1 architecture
    model_module = importlib.import_module(f"models.{args.model_module}.{args.model_module}")
    model_class = getattr(model_module, args.model_module)
    model_folder = os.path.dirname(model_module.__file__)

    # Add input dimensions to model_kwargs
    model_kwargs["node_input_dim"] = train_loader.dataset[0].x.shape[1]
    model_kwargs["edge_input_dim"] = train_loader.dataset[0].edge_attr.shape[1]

    if args.model_module == "cvx":
        all_data = []
        for d in dataloaders.values():
            if d: all_data.extend(d.dataset)
        max_n = max(d.cvx_node_mask.shape[1] for d in all_data)
        max_e = max(d.cvx_edge_mask.shape[1] for d in all_data)
        model_kwargs['max_n'], model_kwargs['max_e'] = max_n, max_e
        model_kwargs['cvx_layer'] = build_cvx_layer(max_n, max_e)

    # Create model with stage 1 architecture
    model = model_class(**model_kwargs).to(device)
    logger.info(f"Created model with architecture from stage 1")

    # 8) Load weights from stage 1
    if model_state_dict is not None:
        try:
            model.load_state_dict(model_state_dict, strict=True)
            logger.info("Successfully loaded stage 1 weights with strict=True")
        except RuntimeError as e:
            logger.warning(f"Could not load weights with strict=True: {e}")
            model.load_state_dict(model_state_dict, strict=False)
            logger.warning("Loaded stage 1 weights with strict=False; some keys may have been ignored")

    # 9) Setup training with stage 2 hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    best_loss = float("inf")
    patience = 0

    logger.info(f"Training {args.model_module} with {len(train_loader.dataset)} samples")
    logger.info(f"Validation with {len(val_loader.dataset)} samples")
    try:
        logger.info(f"Test with {len(test_loader.dataset)} samples")
    except AttributeError:
        logger.info("No test dataset provided")

    # Use criterion from stage 2 hyperparameters
    if args.criterion_name == "FocalLoss":
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    elif args.criterion_name == "WeightedBCELoss":
        criterion = WeightedBCELoss(pos_weight=2.0)
    elif args.criterion_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif args.criterion_name == "MSELoss":
        criterion = nn.MSELoss()
    else:
        try:
            criterion = getattr(nn, args.criterion_name)()
            logger.info(f"Initialized criterion '{args.criterion_name}' from torch.nn")
        except AttributeError:
            logger.error(f"Criterion '{args.criterion_name}' not found in custom losses or torch.nn!")
            raise
    
    # Use lambda values from stage 2
    lambda_dict = {
        'lambda_phy_loss': getattr(args, 'lambda_phy_loss', 0.1),
        'lambda_mask': getattr(args, 'lambda_mask', 0.01),
        'lambda_connectivity': getattr(args, 'lambda_connectivity', 0.05),
        'lambda_radiality': getattr(args, 'lambda_radiality', 0.05),
        'loss_scaling_strategy': getattr(args, 'loss_scaling_strategy', 'adaptive_ratio'),
        'normalization_type': getattr(args, 'normalization_type', 'none'),
    }
    
    writer = SummaryWriter(log_dir="runs/grad_debug")
    global_step = 0 
    
    for epoch in range(args.epochs):
        train_loss, train_dict = train(model, train_loader, optimizer, criterion, device, **lambda_dict, writer=writer, global_step=global_step)
        val_loss, val_dict = test(model, val_loader, criterion, device, **lambda_dict)
        total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Total Grad Norm: {total_grad_norm:.6f}")

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            logger.info(f"\n Epoch {epoch+1}/{args.epochs} - Train Metrics: {train_dict}\n ")
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Val Metrics: {val_dict} \n ")
            
        if args.wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, **train_dict, **val_dict})
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': final_config,
                'stage1_architecture': stage1_architecture,
                'stage2_hyperparams': stage2_hyperparams,
                'epoch': epoch,
                'best_loss': best_loss
            }, f"model_search/models/{args.model_module}/{args.job_name}-Best.pt")
        else:
            patience += 1
            if patience == args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    if test_loader:
        test_loss, test_dict = test(model, test_loader, criterion, device, **lambda_dict) 
        logger.info(f"Test Loss: {test_loss:.4f}")
        if args.wandb:
            wandb.log({"test_loss": test_loss, **test_dict}) 
    logger.info(f"Training completed. Best validation loss: {best_loss:.4f}")

    model_save_path = f"model_search/models/{args.model_module}/{args.job_name}-Epoch{epoch}-Last.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': final_config,
        'stage1_architecture': stage1_architecture,
        'stage2_hyperparams': stage2_hyperparams,
        'epoch': epoch,
        'final_loss': val_loss
    }, model_save_path)
    logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
   main()