import argparse
import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import wandb
import importlib
import sys
import logging
import random
import numpy as np

# Setup logging
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

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "model_search"), str(ROOT_DIR / "src")])

from load_data import create_data_loaders
from train import train, test
from loss_functions import WeightedBCELoss, FocalLoss

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set all random seeds to: {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN models")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--load_checkpoint", type=str, help="Path to checkpoint to load")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    return parser.parse_args()

def load_config(filepath):
    """Load config from YAML file"""
    config_path = Path(filepath)
    if not config_path.is_absolute():
        # Try relative to config_files directory
        config_path = Path("model_search/config_files") / filepath
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert to namespace for compatibility
    return argparse.Namespace(**config)

def convert_hpo_config_to_model_kwargs(args):
    """
    Convert HPO-style config (all params at top level) to main.py style (model_kwargs).
    This handles configs from HPO that don't have model_kwargs structure.
    """
    # If model_kwargs already exists, just ensure required fields are there
    if hasattr(args, 'model_kwargs') and args.model_kwargs:
        return args
    
    # Parameters that should be in model_kwargs for the model
    model_param_keys = [
        'output_type', 'num_classes', 'gnn_type', 'gnn_layers', 'gnn_hidden_dim',
        'gat_heads', 'gat_dropout', 'gin_eps', 'gin_hidden_dim', 'gin_layers',
        'gin_mlp_layers', 'gin_train_eps', 'use_node_mlp', 'node_hidden_dims',
        'use_edge_mlp', 'edge_hidden_dims', 'activation', 'dropout_rate',
        'use_batch_norm', 'use_residual', 'use_skip_connections', 'pooling',
        'switch_head_type', 'switch_head_layers', 'switch_attention_heads',
        'switch_dim_per_head', 'use_gated_mp', 'use_phyr', 'phyr_k_ratio',
        'enforce_radiality'
    ]
    
    # Create model_kwargs from top-level parameters
    args.model_kwargs = {}
    for key in model_param_keys:
        if hasattr(args, key):
            args.model_kwargs[key] = getattr(args, key)
    
    return args

def initialize_criterion(criterion_name):
    """Initialize loss criterion"""
    if criterion_name == "FocalLoss":
        return FocalLoss(alpha=1.0, gamma=2.0)
    elif criterion_name == "WeightedBCELoss":
        return WeightedBCELoss(pos_weight=2.0)
    elif criterion_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif criterion_name == "MSELoss":
        return nn.MSELoss()
    else:
        try:
            return getattr(nn, criterion_name)()
        except AttributeError:
            logger.error(f"Criterion '{criterion_name}' not found!")
            raise

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Convert HPO config format if needed
    config = convert_hpo_config_to_model_kwargs(config)
    
    # Set seed for reproducibility
    seed = getattr(config, 'seed', 0)
    set_seed(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize W&B if requested
    if args.wandb or getattr(config, 'wandb', False):
        run = wandb.init(
            project=getattr(config, 'wandb_project', 'GNN_Training'),
            name=getattr(config, 'job_name', 'default_job'),
            config=vars(config)
        )
        config.job_name = run.name
    
    # Load data
    logger.info("Loading datasets...")
    dataloaders = create_data_loaders(
        dataset_names=config.dataset_names,
        folder_names=config.folder_names,
        dataset_type=getattr(config, 'dataset_type', 'default'),
        batch_size=config.batch_size,
        max_nodes=getattr(config, 'max_nodes', 20000),
        max_edges=getattr(config, 'max_edges', 50000),
        train_ratio=getattr(config, 'train_ratio', 0.85),
        seed=seed,
        num_workers=getattr(config, 'num_workers', 0),
        batching_type="standard",
        shuffle=True,
    )
    
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("validation")
    test_loader = dataloaders.get("test")
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Get input dimensions
    sample_data = train_loader.dataset[0]
    node_input_dim = sample_data.x.shape[1]
    edge_input_dim = sample_data.edge_attr.shape[1]
    
    # Initialize model
    model_module_name = config.model_module
    model_module = importlib.import_module(f"models.{model_module_name}.{model_module_name}")
    model_class = getattr(model_module, model_module_name)
    
    # Prepare model kwargs
    model_kwargs = config.model_kwargs.copy()
    model_kwargs['node_input_dim'] = node_input_dim
    model_kwargs['edge_input_dim'] = edge_input_dim
    
    logger.info(f"Initializing {model_module_name} model...")
    model = model_class(**model_kwargs).to(device)
    
    # Load checkpoint if provided
    if args.load_checkpoint:
        logger.info(f"Loading checkpoint from: {args.load_checkpoint}")
        model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=getattr(config, 'weight_decay', 1e-5)
    )
    
    # Initialize criterion
    criterion = initialize_criterion(config.criterion_name)
    
    # Prepare lambda dictionary for physics losses
    lambda_dict = {
        'lambda_phy_loss': getattr(config, 'lambda_phy_loss', 0.1),
        'lambda_mask': getattr(config, 'lambda_mask', 0.01),
        'lambda_connectivity': getattr(config, 'lambda_connectivity', 0.05),
        'lambda_radiality': getattr(config, 'lambda_radiality', 0.05),
        'loss_scaling_strategy': getattr(config, 'loss_scaling_strategy', 'adaptive_ratio'),
        'normalization_type': getattr(config, 'normalization_type', 'adaptive'),
    }
    
    logger.info(f"Physics loss configuration: {lambda_dict}")
    
    # Training loop
    best_val_loss = float('inf')
    best_val_mcc = -1.0
    patience_counter = 0
    patience = getattr(config, 'patience', 10)
    epochs = getattr(config, 'epochs', 100)
    
    # Create save directory
    save_dir = Path(f"model_search/models/{model_module_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"config {config}")    
    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_loss, train_metrics = train(
            model, train_loader, optimizer, criterion, device, **lambda_dict
        )
        
        # Validate
        val_loss, val_metrics = test(
            model, val_loader, criterion, device, **lambda_dict
        )
        
        # Get MCC for model selection
        val_mcc = val_metrics.get('test_mcc', -1.0)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Val MCC: {val_mcc:.4f}")
        
        # Log detailed metrics every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f"Train metrics: {train_metrics}")
            logger.info(f"Val metrics: {val_metrics}")
        
        # W&B logging
        if args.wandb or getattr(config, 'wandb', False):
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mcc": val_mcc,
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })
        
        # Save best model based on MCC
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = save_dir / f"{config.job_name}_best_mcc.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mcc': val_mcc,
                'config': vars(config)
            }, best_model_path)
            logger.info(f"Saved best model (MCC: {val_mcc:.4f}) to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Test evaluation
    if test_loader:
        logger.info("\nEvaluating on test set...")
        # Load best model
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_metrics = test(
            model, test_loader, criterion, device, **lambda_dict
        )
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test MCC: {test_metrics.get('test_mcc', -1.0):.4f}")
        logger.info(f"Test metrics: {test_metrics}")
        
        if args.wandb or getattr(config, 'wandb', False):
            wandb.log({
                "test_loss": test_loss,
                "test_mcc": test_metrics.get('test_mcc', -1.0),
                **{f"test_{k}": v for k, v in test_metrics.items()}
            })
    
    # Save final model
    final_model_path = save_dir / f"{config.job_name}_final.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(config)
    }, final_model_path)
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation MCC: {best_val_mcc:.4f}")
    logger.info(f"Models saved in: {save_dir}")
    
    if args.wandb or getattr(config, 'wandb', False):
        wandb.finish()

if __name__ == "__main__":
    main()