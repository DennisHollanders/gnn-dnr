import argparse
import os
from sqlalchemy import FallbackAsyncAdaptedQueuePool
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import wandb
import importlib
import sys
import logging

import numpy as np
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

MODEL_KWARGS_KEYS = [ "output_type", "num_classes",
    "gnn_type", "gnn_layers", "gnn_hidden_dim", "gat_heads", "dropout_rate",
    "gat_dropout", "gin_eps","gin_mlp_layers", "gin_train_eps","gat_v2","gat_edge_dim",
    "phyr_k_ratio",
    "node_hidden_dims", "use_node_mlp", "edge_hidden_dims", "use_edge_mlp",
    "activation", "use_batch_norm", "use_residual", "use_skip_connections",
    "pooling", "switch_head_type", "switch_head_layers", "switch_attention_heads",
    "use_gated_mp", "use_phyr", "enforce_radiality",

]

STAGE1_KEYS = [
    "activation", "use_residual", "use_skip_connections", "use_gated_mp",
    "gnn_layers", "gnn_hidden_dim", "gat_heads",  "gat_dropout", "gin_mlp_layers", "gin_train_eps","gat_v2","gat_edge_dim",
    "phyr_k_ratio",
    "node_hidden_dims", "use_node_mlp", "edge_hidden_dims", "use_edge_mlp",
    "switch_head_type", "switch_head_layers", "switch_attention_heads"
]

LAMBDAS= [
    "lambda_phy_loss", "lambda_mask", "lambda_connectivity", "lambda_radiality"]
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
def load_and_merge_checkpoints(stage1_config, stage2_config, stage2_seed=None):
    # Start with stage2 config as base
    combined_config = dict(stage2_config)

    # Override architecture keys from stage1
    for k in STAGE1_KEYS:
        if k in stage1_config:
            combined_config[k] = stage1_config[k]
    combined_config["seed"] = stage2_seed
    # Create Namespace
    vargs = argparse.Namespace(**combined_config)

    # Build model_kwargs from MODEL_KWARGS_KEYS
    model_kwargs = {}
    for k in MODEL_KWARGS_KEYS:
        if k in stage1_config:
            model_kwargs[k] = stage1_config[k]
        elif hasattr(vargs, k):
            model_kwargs[k] = getattr(vargs, k)

    vargs.model_kwargs = model_kwargs

    # Remove keys from top-level if they belong in model_kwargs
    for k in MODEL_KWARGS_KEYS:
        if hasattr(vargs, k):
            delattr(vargs, k)

    # Set default model_module
    if not hasattr(vargs, "model_module"):
        vargs.model_module = stage1_config.get("model_module", "AdvancedMLP")

    # Ensure model_path exists
    if not hasattr(vargs, "model_path"):
        vargs.model_path = None

    return vargs
def main():

    args = parse_args()
    stage1_hyperparams = {}
    stage2_hyperparams = {}


    print("args before loading from yaml:", args)
    if args.config:
        logger.info(f"Loaded configuration from {args.config}")
        args = load_args_from_yaml(args.config)
    elif args.stage1_checkpoint and args.stage2_checkpoint:
        logger.info(f"Loading stage 1 and stage 2 checkpoints: {args.stage1_checkpoint}, {args.stage2_checkpoint}")
        stage2_checkpoint_path = args.stage2_checkpoint
        stage2_checkpoint = torch.load(stage2_checkpoint_path, map_location="cpu")
        stage2_config = stage2_checkpoint.get("config", {})
        stage2_seed = stage2_checkpoint.get("seed", None)
        print(f"\n \n =================================Stage 2 configs: {stage2_config}")

        logger.info(f"Loading stage 1 architecture from {args.stage1_checkpoint}")
        stage1_checkpoint_path = args.stage1_checkpoint
        stage1_checkpoint = torch.load(stage1_checkpoint_path, map_location="cpu")
        stage1_config = stage1_checkpoint.get("config", {})
        print(f"Stage 1 configs: {stage1_config} \n \n ==================================="   )
        # print lambda values from both stages
        print(f"stage 1 lambda: {[stage1_config.get(k) for k in LAMBDAS]}")
        print(f"stage 2 lambda: {[stage2_config.get(k) for k in LAMBDAS]}")
        args = load_and_merge_checkpoints(stage1_config, stage2_config, stage2_seed)  
        args.stage1_checkpoint = stage1_checkpoint_path
        args.stage2_checkpoint = stage2_checkpoint_path
        args.model_module = "AdvancedMLP"  
        args.seed =stage1_checkpoint["seed"]

    args.folder_names = ["data/split_datasets/train", "data/split_datasets/validation", "data/split_datasets/test"  ]
    #args.folder_names = ["data/split_datasets-without-synthetic/train", "data/split_datasets-without-synthetic/validation", "data/split_datasets-without-synthetic/test"]
    import random
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enforce deterministic algorithms in PyTorch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    final_config = vars(args)

    # Print with YAML formatting
    print("========== FINAL CONFIG STRUCTURE ==========")
    print(yaml.dump(final_config, sort_keys=False, default_flow_style=False))


    print("args after loading from yaml:", args)
    model_module = importlib.import_module(f"models.{args.model_module}.{args.model_module}")
    model_folder = os.path.dirname(model_module.__file__)
    print("DEBUG: Value of args.model_module before import:", args.model_module)

    # Get the model class from the imported module
    model_class = getattr(model_module, args.model_module)
    print(f"Successfully imported model: {args.model_module}")

    args.description = "Ablation-studies"
    run = wandb.init(project=args.description, job_type="train", config=vars(args))
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
        num_workers= 4, 
        batching_type=args.batching_type,
        shuffle=True,  # Set shuffle to True for training data
    )
    train_loader, val_loader, test_loader = dataloaders.get("train"),  dataloaders.get("validation"), dataloaders.get("test")

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

    if args.model_module == "cvx":
        all_data = []
        for d in dataloaders.values():
            if d: all_data.extend(d.dataset)

        max_n_val = max(d.cvx_node_mask.shape[1] for d in all_data)
        max_e_val = max(d.cvx_edge_mask.shape[1] for d in all_data)
        model_kwargs['max_n'] = max_n_val
        model_kwargs['max_e'] = max_e_val
        model_kwargs['cvx_layer'] = build_cvx_layer(max_n_val, max_e_val)

        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        # enforce a sane DataLoader num_workers
        args.num_workers = min(args.num_workers, 4)
    print("Loaded model_kwargs: =================================================================")
    for k, v in args.model_kwargs.items():
        print(f"{k}: {v}")

    model = model_class(**model_kwargs)
    if args.stage1_checkpoint:
        logger.info(f"Loading stage 1 checkpoint from {args.stage1_checkpoint}")
        stage1_checkpoint = torch.load(args.stage1_checkpoint, map_location="cpu")
        model.load_state_dict(stage1_checkpoint['initial_model_state_dict'], strict=True)
  
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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

    lambda_dict = {
        'lambda_phy_loss': getattr(args, 'lambda_phy_loss', 0.1),
        'lambda_mask': getattr(args, 'lambda_mask', 0.01),
        'lambda_connectivity': getattr(args, 'lambda_connectivity', 0.05),
        'lambda_radiality': getattr(args, 'lambda_radiality', 0.05),
    }

    
    writer = SummaryWriter(log_dir="runs/grad_debug")


    logger.info(f"Python hash seed: {os.environ.get('PYTHONHASHSEED', 'not set')}")
    logger.info(f"Torch deterministic: {torch.backends.cudnn.deterministic}")
    logger.info(f"Torch benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"NumPy random state: {np.random.get_state()[1][0]}") 

    best_loss = float('inf')
    patience = 0
    global_step = 0 

    for epoch in range(300):
        if epoch == 0:
            logger.info(f"Training config - LR: {args.learning_rate}, "
                f"WD: {args.weight_decay}, Batch size: {args.batch_size}, "
                f"Criterion: {args.criterion_name}")
            logger.info(f"Lambda values: {lambda_dict}")
        train_loss, train_dict = train(model, train_loader, optimizer, criterion, device, **lambda_dict, writer=writer, global_step=global_step)
        val_loss, val_dict = test(model, val_loader, criterion, device, **lambda_dict)
        total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Total Grad Norm: {total_grad_norm:.6f}")

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            logger.info(f"\n Epoch {epoch+1}/{args.epochs} - Train Metrics: {train_dict}\n ")
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Val Metrics: {val_dict} \n ")
            
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, **train_dict, **val_dict})
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': final_config,
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
        logger.info(f"Test Metrics: {test_dict}")
        if args.wandb:
            wandb.log({"test_loss": test_loss, **test_dict}) 
    logger.info(f"Training completed. Best validation loss: {best_loss:.4f}")

    model_save_path = f"model_search/models/{args.model_module}/{args.job_name}-Epoch{epoch}-Last.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': final_config,
        'epoch': epoch,
        'final_loss': val_loss
    }, model_save_path)
    logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
   main()