import argparse
from typing import final
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
import tempfile
import os
from datetime import datetime
import pickle
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import csv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("HPO")

# Add paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "model_search")])

from load_data import create_data_loaders
from train import train, test
# Add paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "src")])

from loss_functions import FocalLoss, WeightedBCELoss
class SimpleHPO:
    """Simplified HPO with feasibility checks, CSV tracking, and W&B experiment grouping"""
    
    def __init__(self, config_path: str, study_name: str, startup_trials: int = 10,
                  pruner_startup: int = 5, pruner_warmup: int = 10):
        self.n_startup_trials = startup_trials
        self.pruner_startup = pruner_startup
        self.pruner_warmup = pruner_warmup
        self.seed = 2
        self.config_path = config_path
        self.study_name = study_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(config_path, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.search_space = self.config.pop('search_space', {})
        self.fixed_params = self.config.pop('fixed_params', {})
        
        # Results directory
        model_name = self.config.get('model_module', 'Unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path(f"model_search/models/{model_name}/hpo_{study_name}_{timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV tracking
        self.csv_file = self.results_dir / "hpo_results.csv"
        self.csv_headers = None
        self.csv_writer = None
        self._init_csv_tracking()
        
        self.wandb_run = None
        self.experiment_id = f"{study_name}_{timestamp}"

        # NEW: Pre-load data once
        self.train_loader = None
        self.val_loader = None
        self.data_sample = None
        self._load_data_once()
    
    def _init_csv_tracking(self):
        """Initialize CSV file for tracking all trial results"""
        # Create CSV with headers for hyperparameters + metrics
        self.csv_file.touch()
        
    def _log_trial_to_csv(self, trial_num: int, config: dict, metrics: dict):
        """Log trial results to CSV with proper handling of list parameters"""
        # Combine all data
        row_data = {
            'trial_number': trial_num,
            'timestamp': datetime.now().isoformat(),
            **metrics  # Put metrics first to avoid column mismatches
        }
        
        # Add config parameters, handling lists properly
        for key, value in config.items():
            if isinstance(value, list):
                # Convert lists to string representation without commas that break CSV
                row_data[key] = str(value).replace(',', ';')
            elif isinstance(value, (dict, tuple)):
                # Handle other complex types
                row_data[key] = str(value).replace(',', ';')
            else:
                row_data[key] = value
        
        # Write to CSV with proper quoting
        df = pd.DataFrame([row_data])
        
        # If file is empty, write headers
        if not self.csv_file.exists() or self.csv_file.stat().st_size == 0:
            df.to_csv(self.csv_file, index=False, quoting=1) 
        else:
            df.to_csv(self.csv_file, mode='a', header=False, index=False, quoting=1)
    
    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Suggest parameters with FIXED feasibility checks using YAML config"""
        config = self.config.copy()
        config.update(self.fixed_params)
        
        # Phase 1: Suggest GAT heads FIRST 
        if 'gat_heads' not in config and 'gat_heads' in self.search_space:
            spec = self.search_space['gat_heads']
            if spec['search_type'] == 'categorical':
                config['gat_heads'] = trial.suggest_categorical('gat_heads', spec['choices'])
            elif spec['search_type'] == 'int':
                config['gat_heads'] = trial.suggest_int('gat_heads', spec['min'], spec['max'])
        
        # Phase 2: Suggest GNN hidden dim with GAT constraint awareness
        if 'gnn_hidden_dim' not in config and 'gnn_hidden_dim' in self.search_space:
            spec = self.search_space['gnn_hidden_dim']
            gat_heads = config.get('gat_heads', 1) 
            
            if spec['search_type'] == 'categorical':
                # Filter choices to be divisible by GAT heads
                valid_choices = [dim for dim in spec['choices'] if dim % gat_heads == 0]
                if not valid_choices:
                    # Fallback: find nearest multiples
                    valid_choices = [((dim // gat_heads) + 1) * gat_heads for dim in spec['choices']]
                    valid_choices = list(set(valid_choices)) 
                config['gnn_hidden_dim'] = trial.suggest_categorical('gnn_hidden_dim', valid_choices)
            elif spec['search_type'] == 'int':
                # Suggest in multiples of gat_heads
                min_val = ((spec['min'] // gat_heads) + (1 if spec['min'] % gat_heads != 0 else 0)) * gat_heads
                max_val = (spec['max'] // gat_heads) * gat_heads
                if min_val > max_val:
                    min_val = max_val = gat_heads
                config['gnn_hidden_dim'] = trial.suggest_int('gnn_hidden_dim_multiple', 
                                                            min_val // gat_heads, 
                                                            max_val // gat_heads) * gat_heads
        
        # Phase 3: Suggest switch head type
        if 'switch_head_type' not in config and 'switch_head_type' in self.search_space:
            spec = self.search_space['switch_head_type']
            config['switch_head_type'] = trial.suggest_categorical('switch_head_type', spec['choices'])
        
        # Phase 4: Suggest attention heads with constraint awareness
        if ('attention' in config.get('switch_head_type', '').lower() and 
            'switch_attention_heads' not in config and 
            'switch_attention_heads' in self.search_space):
            
            spec = self.search_space['switch_attention_heads']
            gnn_hidden_dim = config.get('gnn_hidden_dim', 128)
            
            if spec['search_type'] == 'categorical':
                valid_choices = [heads for heads in spec['choices'] if gnn_hidden_dim % heads == 0]
                if not valid_choices:
                    # Find factors of gnn_hidden_dim within the choice range
                    valid_choices = []
                    for heads in spec['choices']:
                        if gnn_hidden_dim >= heads and gnn_hidden_dim % heads == 0:
                            valid_choices.append(heads)
                    if not valid_choices:
                        valid_choices = [1]  
                config['switch_attention_heads'] = trial.suggest_categorical('switch_attention_heads', valid_choices)
            elif spec['search_type'] == 'int':
                # Find valid divisors within range
                valid_heads = []
                for heads in range(spec['min'], spec['max'] + 1):
                    if gnn_hidden_dim % heads == 0:
                        valid_heads.append(heads)
                if not valid_heads:
                    valid_heads = [1]  
                config['switch_attention_heads'] = trial.suggest_categorical('switch_attention_heads_valid', valid_heads)
        
        # Phase 5: Process all other parameters
        for param, spec in self.search_space.items():
            if param in config:
                continue  # Already handled above
                
            if spec['search_type'] == 'float':
                config[param] = trial.suggest_float(
                    param, spec['min'], spec['max'], 
                    log=spec.get('log', False)
                )
            elif spec['search_type'] == 'int':
                config[param] = trial.suggest_int(
                    param, spec['min'], spec['max'], 
                    step=spec.get('step', 1)
                )
            elif spec['search_type'] == 'categorical':
                config[param] = trial.suggest_categorical(param, spec['choices'])
            elif spec['search_type'] == 'dynamic_list':
                config[param] = self._suggest_dynamic_list(trial, param, spec)
        
        # Phase 6: Final constraint validation and fixes
        config = self._apply_constraints(trial, config)
        
        return config
    
    def _suggest_dynamic_list(self, trial: optuna.Trial, param: str, spec: dict) -> list:
        """Handle dynamic list parameters like hidden_dims - constrained to powers of 2"""
        n_layers = trial.suggest_int(f'n_{param}_layers', 
                                   spec.get('n_layers_min', 1), 
                                   spec.get('n_layers_max', 3))
        
        # Generate valid powers of 2 within the specified range
        dim_min = spec.get('dim_min', 32)
        dim_max = spec.get('dim_max', 1024)

        # Find powers of 2 within range
        valid_powers = []
        power = 1
        while power <= dim_max:
            if power >= dim_min:
                valid_powers.append(power)
            power *= 2
        
        # Fallback if no valid powers found
        if not valid_powers:
            valid_powers = [32, 64, 128, 256, 512, 1024]  
            valid_powers = [p for p in valid_powers if dim_min <= p <= dim_max]
            if not valid_powers:
                valid_powers = [dim_min] 
        
        dims = []
        for i in range(n_layers):
            dim = trial.suggest_categorical(f'{param}_{i}', valid_powers)
            dims.append(dim)
        
        return dims
    
    def _optuna_constraints(self, trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
        """Constraint function for Optuna sampler"""
        p = trial.params
        cons = []
        
        # Constraint 1: GAT divisibility
        if p.get('gnn_type') == 'GAT':
            gnn_hidden_dim = p.get('gnn_hidden_dim')
            gat_heads = p.get('gat_heads')
            if gnn_hidden_dim is not None and gat_heads is not None:
                remainder = gnn_hidden_dim % gat_heads
                cons.append(float(remainder)) 
        
        # Constraint 2: Switch attention heads divisibility
        switch_head_type = p.get('switch_head_type', '')
        if 'attention' in switch_head_type.lower():
            gnn_hidden_dim = p.get('gnn_hidden_dim')
            switch_heads = p.get('switch_attention_heads')
            if gnn_hidden_dim is not None and switch_heads is not None:
                cons.append(float(gnn_hidden_dim % switch_heads))
        
        # Constraint 3: At least one MLP must be enabled
        use_node_mlp = p.get('use_node_mlp', True)
        use_edge_mlp = p.get('use_edge_mlp', True)
        if not use_node_mlp and not use_edge_mlp:
            cons.append(1.0)  
        else:
            cons.append(0.0) 
        
        return tuple(cons)
    
    def _apply_constraints(self, trial: optuna.Trial, config: dict) -> dict:
        """Apply comprehensive feasibility constraints"""
        
        # Constraint 1: gnn_hidden_dim % gat_heads == 0 (for GAT)
        if config.get('gnn_type') == 'GAT':
            gnn_hidden_dim = config.get('gnn_hidden_dim', 64)
            gat_heads = config.get('gat_heads', 4)
            
            # Make gnn_hidden_dim divisible by gat_heads
            remainder = gnn_hidden_dim % gat_heads
            if remainder != 0:
                config['gnn_hidden_dim'] = gnn_hidden_dim + (gat_heads - remainder)
                logger.debug(f"Adjusted gnn_hidden_dim to {config['gnn_hidden_dim']} for GAT heads")
        
        # Constraint 2: hidden_dim % gat_heads == 0 (if hidden_dim exists)
        if config.get('gnn_type') == 'GAT' and 'hidden_dim' in config:
            hidden_dim = config['hidden_dim']
            gat_heads = config.get('gat_heads', 4)
            
            remainder = hidden_dim % gat_heads
            if remainder != 0:
                config['hidden_dim'] = hidden_dim + (gat_heads - remainder)
        
        # Constraint 3: switch_attention_heads divisibility 
        if 'switch_attention_heads' in config and 'hidden_dim' in config:
            hidden_dim = config['hidden_dim']
            switch_heads = config['switch_attention_heads']
            
            remainder = hidden_dim % switch_heads
            if remainder != 0:
                config['hidden_dim'] = hidden_dim + (switch_heads - remainder)
        
        # Constraint 4: Conditional parameters for GAT
        if config.get('gnn_type') != 'GAT':
            # Remove GAT-specific parameters if not using GAT
            config.pop('gat_heads', None)
            config.pop('gat_dropout', None)
            config.pop('gat_v2', None)
            config.pop('gat_edge_dim', None)
        
        # Constraint 5: Conditional parameters for GIN
        if config.get('gnn_type') != 'GIN':
            # Remove GIN-specific parameters if not using GIN
            config.pop('gin_eps', None)
            config.pop('gin_train_eps', None)
            config.pop('gin_mlp_layers', None)
        
        # Constraint 6: Switch attention heads only for attention-based switch heads
        switch_head_type = config.get('switch_head_type', 'mlp')
        if 'attention' not in switch_head_type.lower():
            config.pop('switch_attention_heads', None)
        
        # Constraint 7: Node MLP dims only if using node MLP
        if not config.get('use_node_mlp', True):
            config.pop('node_hidden_dims', None)
            
        # Constraint 8: Edge MLP dims only if using edge MLP  
        if not config.get('use_edge_mlp', True):
            config.pop('edge_hidden_dims', None)
        
        # Constraint 9: Ensure at least one MLP is used
        if not config.get('use_node_mlp') and not config.get('use_edge_mlp'):
            config['use_node_mlp'] = True
            if 'node_hidden_dims' not in config:
                config['node_hidden_dims'] = [128]
        
        # Constraint 10: Gated MP requires GNN
        if config.get('use_gated_mp') and not config.get('gnn_type'):
            config['use_gated_mp'] = False
            logger.debug("Disabled gated MP because no GNN type specified")
        
        # Constraint 11: PhyR parameters only if using PhyR
        if not config.get('use_phyr', False):
            config.pop('phyr_k_ratio', None)
        
        # Constraint 12: Match criterion to output type
        output_type = config.get('output_type', 'multiclass')
        if output_type == 'multiclass' and 'criterion_name' not in self.fixed_params:
            config['criterion_name'] = "WeightedBCELoss"
        elif output_type == 'binary' and 'criterion_name' not in self.fixed_params:
            config['criterion_name'] = 'BCEWithLogitsLoss'
        elif output_type == 'regression' and 'criterion_name' not in self.fixed_params:
            config['criterion_name'] = 'MSELoss'
    
        
        # Constraint 14: Memory limit (batch_size × max_nodes)
        batch_size = config.get('batch_size', 32)
        max_nodes = config.get('max_nodes', 1000)
        memory_limit = 100000  
        
        if batch_size * max_nodes > memory_limit:
            # Reduce batch size to fit memory
            config['batch_size'] = max(1, memory_limit // max_nodes)
            logger.debug(f"Reduced batch_size to {config['batch_size']} for memory constraint")
        
        return config
    
    def _validate_config(self, config: dict) -> bool:
        """Validate configuration before trial execution"""
        
        # Check divisibility constraints
        if config.get('gnn_type') == 'GAT':
            gnn_hidden_dim = config.get('gnn_hidden_dim', 64)
            gat_heads = config.get('gat_heads', 4)
            
            if gnn_hidden_dim % gat_heads != 0:
                logger.warning(f"Invalid GAT config: {gnn_hidden_dim} not divisible by {gat_heads}")
                return False
        
        # Check that at least one MLP is enabled
        if not config.get('use_node_mlp') and not config.get('use_edge_mlp'):
            logger.warning("Invalid config: Neither node nor edge MLP enabled")
            return False
        
        # Check logical dependencies
        if config.get('use_gated_mp') and not config.get('gnn_type'):
            logger.warning("Invalid config: Gated MP requires GNN")
            return False
        
        # Check PhyR dependency
        if config.get('use_phyr') and 'phyr_k_ratio' not in config:
            logger.warning("Invalid config: PhyR enabled but no k_ratio specified")
            return False
        
        # Check conditional parameters
        if config.get('gnn_type') != 'GAT' and any(k in config for k in ['gat_heads', 'gat_dropout']):
            logger.warning("Invalid config: GAT parameters with non-GAT backbone")
            return False
            
        if config.get('gnn_type') != 'GIN' and 'gin_eps' in config:
            logger.warning("Invalid config: GIN parameters with non-GIN backbone")
            return False
        
        return True
    
    def objective(self, trial: optuna.Trial) -> float:
        trial_run = None
        if self.wandb_run:
            trial_run = wandb.init(
                project=self.config.get('wandb_project', 'HPO'),
                group=self.experiment_id, 
                name=f"trial_{trial.number:03d}",
                job_type="hpo_trial",
                tags=[self.study_name, "hpo_trial"],
                config=None, 
                reinit=True
            )
        
        try:
            config = self.suggest_params(trial)
            
            if not self._validate_config(config):
                logger.warning(f"Trial {trial.number}: Invalid configuration, pruning")
                if trial_run:
                    wandb.log({"status": "invalid_config"})
                    wandb.finish()
                raise optuna.TrialPruned()
            
            if trial_run:
                wandb.config.update(config)
            
            # Create model and data
            model, train_loader, val_loader, criterion = self._setup_training(config)
            if model is None:
                if trial_run:
                    wandb.log({"status": "failed_setup"})
                    wandb.finish()
                return -1.0 
            
            model_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Trial {trial.number}: Config valid, Model params: {model_params:,}")
            
            optimizer = optim.Adam(model.parameters(), 
                                lr=config['learning_rate'], 
                                weight_decay=config['weight_decay'])

            lambda_dict = {
                'lambda_phy_loss': config.get('lambda_phy_loss', 0.1),
                'lambda_mask': config.get('lambda_mask', 0.01),
                'lambda_connectivity': config.get('lambda_connectivity', 0.05),
                'lambda_radiality': config.get('lambda_radiality', 0.05),
                'normalization_type': config.get('normalization_type', 'adaptive'),
                'loss_scaling_strategy': config.get('loss_scaling_strategy', 'adaptive_ratio')
            }
            
            starting_val_loss = float('inf')
            best_train_loss = float('inf')
            best_val_loss = float('inf')
            best_minority_f1 = 0.0
            best_mcc = -1.0 
            best_balanced_accuracy = 0.0
        
            patience = 0
            max_epochs = min(config.get('epochs', 100), 80)
            max_patience = config.get('patience', 25)
            
            print(f"Starting training for {max_epochs} epochs with patience {max_patience}")
            print("config:", config)
            print(f"Lambda dict: {lambda_dict}")  
            
            for epoch in range(max_epochs):
                train_loss, train_dict = train(model, train_loader, optimizer, criterion, self.device, **lambda_dict)
                val_loss, val_dict = test(model, val_loader, criterion, self.device, **lambda_dict)
                
                if epoch == 0: 
                    starting_val_loss = val_loss
                    starting_train_loss = train_loss
                    
                print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                if epoch % 5 == 0 or epoch ==0:
                    print(f"\n Epoch {epoch+1}- Train Metrics: {train_dict}\n ")
                    print(f"Epoch {epoch+1} - Val Metrics: {val_dict} \n ")
                    

                current_f1_minority = val_dict.get('test_f1_minority', 0.0)
                current_mcc = val_dict.get('test_mcc', 0.0)
                current_balanced_accuracy = val_dict.get('test_balanced_acc', 0.0)
                
                if current_mcc > best_mcc:
                    best_mcc = current_mcc
                    best_minority_f1 = current_f1_minority
                    best_balanced_accuracy = current_balanced_accuracy
                    best_train_loss = train_loss
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if trial_run:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_f1_minority": current_f1_minority,
                        "val_mcc": current_mcc,
                        "val_balanced_accuracy": current_balanced_accuracy,
                        "patience": patience,
                        **lambda_dict  
                    })
                
                trial.report(current_mcc, epoch)
                if trial.should_prune() or patience >= max_patience:
                    break

            metrics = {
                "starting_val_loss": starting_val_loss,
                "starting_train_loss": starting_train_loss,
                'best_mcc': best_mcc,  
                'best_train_loss': best_train_loss,
                'best_val_loss': best_val_loss,
                "best_f1_minority": best_minority_f1,
                "best_balanced_accuracy": best_balanced_accuracy,
                "final_train_loss": train_loss,
                "final_val_loss" : val_loss,
                "final_f1_minority" : current_f1_minority,
                "final_mcc" : current_mcc,
                "final_balanced_accuracy" : current_balanced_accuracy,
                'final_epoch': epoch,
                'converged': patience < max_patience,
                'model_parameters': model_params,
                'status': 'completed',
                'config_valid': True
            }
            
            for key, value in metrics.items():
                trial.set_user_attr(key, value)

            if trial_run:
                wandb.log(metrics)
                wandb.log({"trial_score": best_mcc})  
                wandb.finish()

            self._log_trial_to_csv(trial.number, config, metrics)
            logger.info(f"Trial {trial.number}: MCC={best_mcc:.4f}, Val_Loss={best_val_loss:.4f}, Epoch={epoch}, Params={model_params:,}")
            
            return best_mcc 
            
        except optuna.TrialPruned:
            pruned_metrics = {
                "starting_val_loss": float('inf'),
                "starting_train_loss": float('inf'),
                'best_mcc': 0.0,  
                "best_f1_minority": 0.0,
                "best_balanced_accuracy": 0.0,
                'best_train_loss': float('inf'),
                'best_val_loss': float('inf'),
                'final_train_loss': float('inf'),
                'final_val_loss': float('inf'),
                'final_f1_minority': 0.0,
                'final_mcc': 0.0,  
                'final_balanced_accuracy': 0.0,
                'final_epoch': 0,
                'converged': False,
                'model_parameters': 0,
                'status': 'pruned',
                'config_valid': True
            }
            
            if trial_run:
                wandb.log(pruned_metrics)
                wandb.finish()
            
            self._log_trial_to_csv(trial.number, self.suggest_params(trial), pruned_metrics)
            logger.info(f"Trial {trial.number}: Pruned")
            raise
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return -1.0
    
    def _load_data_once(self):
        """Load data once during initialization to avoid repeated loading"""
        try:
            logger.info("Loading data once for all trials...")

            dataloaders = create_data_loaders(
                dataset_names=self.config['dataset_names'],
                folder_names=self.config['folder_names'],
                dataset_type=self.config.get('dataset_type', 'default'),
                batch_size=self.config['batch_size'],
                max_nodes=self.config.get('max_nodes', 1000),
                max_edges=self.config.get('max_edges', 5000),
                train_ratio=self.config.get('train_ratio', 0.85),
                seed=self.seed,
                num_workers=self.config.get('num_workers', 0),
                batching_type=self.config.get('batching_type', 'dynamic'),
            )
            
            self.train_loader = dataloaders.get("train")
            self.val_loader = dataloaders.get("validation")
            
            if not self.train_loader or not self.val_loader:
                raise ValueError("Failed to create data loaders")
            
            self.data_sample = self.train_loader.dataset[0]
            
            logger.info(f"Data loaded successfully: {len(self.train_loader)} train batches, {len(self.val_loader)} val batches")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _create_dynamic_data_loader(self, batch_size: int):
        """Create new data loaders with different batch size if needed"""
        if batch_size == self.config['batch_size']:
            return self.train_loader, self.val_loader
        
        # Create new loaders with different batch size
        dataloaders = create_data_loaders(
            dataset_names=self.config['dataset_names'],
            folder_names=self.config['folder_names'],
            dataset_type=self.config.get('dataset_type', 'default'),
            batch_size=batch_size,  
            max_nodes=self.config.get('max_nodes', 1000),
            max_edges=self.config.get('max_edges', 5000),
            train_ratio=self.config.get('train_ratio', 0.85),
            seed=self.seed,
            num_workers=self.config.get('num_workers', 0),
            batching_type=self.config.get('batching_type', 'dynamic'),
        )
        
        return dataloaders.get("train"), dataloaders.get("validation")

    def _setup_training(self, config: dict):
        """Setup model and criterion (data is already loaded)"""
        try:
            # Load model
            model_module = importlib.import_module(f"models.{config['model_module']}.{config['model_module']}")
            model_class = getattr(model_module, config['model_module'])
            
            # Get data loaders 
            train_loader, val_loader = self._create_dynamic_data_loader(config['batch_size'])
            
            if not train_loader or not val_loader:
                return None, None, None, None

            model_kwargs = {
                'node_input_dim': self.data_sample.x.shape[1],
                'edge_input_dim': self.data_sample.edge_attr.shape[1],
            }

            # FIXED: Include ALL possible model parameters
            for key in ['activation', 'dropout_rate', 'gnn_type', 'gnn_layers', 'gnn_hidden_dim', 
                    'gat_heads', 'gat_dropout', 'gin_eps', 'use_node_mlp', 'use_edge_mlp',
                    'node_hidden_dims', 'edge_hidden_dims', 'use_batch_norm', 'use_residual',
                    'use_skip_connections', 'switch_head_type', 'switch_head_layers', 
                    'switch_attention_heads', 'output_type', 'num_classes', 'use_gated_mp',
                    'use_phyr', 'phyr_k_ratio', 'pooling', 'normalization_type', 
                    'loss_scaling_strategy', 'enforce_radiality']:  # Added missing parameters
                if key in config:
                    model_kwargs[key] = config[key]

            # Handle the node/edge MLP conversion that was causing issues
            if 'node_mlp_layers' in config and 'node_mlp_dim' in config:
                config['node_hidden_dims'] = [config['node_mlp_dim']] * config['node_mlp_layers']
                model_kwargs['node_hidden_dims'] = config['node_hidden_dims']
            
            if 'edge_mlp_layers' in config and 'edge_mlp_dim' in config:
                config['edge_hidden_dims'] = [config['edge_mlp_dim']] * config['edge_mlp_layers']
                model_kwargs['edge_hidden_dims'] = config['edge_hidden_dims']
            
            model = model_class(**model_kwargs).to(self.device)

            criterion_name = config.get('criterion_name', 'MSELoss')
            if criterion_name == "WeightedBCELoss":
                criterion = WeightedBCELoss(pos_weight=2.0)
            elif criterion_name == "FocalLoss":
                criterion = FocalLoss(alpha=1.0, gamma=2.0)
            elif criterion_name == "MSELoss":
                criterion = nn.MSELoss()
            elif criterion_name == "CrossEntropyLoss":
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = getattr(nn, criterion_name)()
            
            return model, train_loader, val_loader, criterion
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return None, None, None, None
    
    def create_parallel_plot(self, study: optuna.Study) -> go.Figure:
        """Create parallel coordinates plot with robust CSV handling"""
        try:
            if self.csv_file.exists() and self.csv_file.stat().st_size > 0:
                try:
                    df = pd.read_csv(self.csv_file, quoting=1, on_bad_lines='skip')
                    df = df[df['status'] == 'completed'].sort_values('best_mcc', ascending=False)  # Changed: Sort by MCC
                    
                    if len(df) >= 5:
                        return self._create_plot_from_dataframe(df)
                except Exception as e:
                    logger.warning(f"Could not read CSV for plotting: {e}")
        except Exception:
            pass
 
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) < 5:
            return None
        

        data = []
        for trial in completed_trials:
            row = trial.params.copy()
            for key, value in row.items():
                if isinstance(value, list):
                    row[key] = str(value)
            row['best_mcc'] = trial.user_attrs.get('best_mcc', trial.value)
            row['Trial'] = trial.number
            data.append(row)
        
        df = pd.DataFrame(data).sort_values('best_mcc', ascending=False)
        return self._create_plot_from_dataframe(df, eval_metric='best_mcc') 
    
    def _create_plot_from_dataframe(self, df: pd.DataFrame, eval_metric: str = 'best_mcc') -> go.Figure:
        """Create parallel coordinates plot from DataFrame"""

        dimensions = [
            dict(label=eval_metric, values=df[eval_metric], 
                 range=[df[eval_metric].min(), df[eval_metric].max()])
        ]
        
        for metric in ['best_val_loss', 'final_epoch']:
            if metric in df.columns and df[metric].notna().any():
                dimensions.append(dict(
                    label=metric.replace('_', ' ').title(),
                    values=df[metric],
                    range=[df[metric].min(), df[metric].max()]
                ))
        
        # Add hyperparameters 
        exclude_cols = {'trial_number', 'timestamp', 'best_f1_score', 'best_train_loss', 
                       'best_val_loss', 'final_train_loss', 'final_val_loss', 'final_precision',
                       'final_recall', 'final_accuracy', 'final_epoch', 'converged', "starting_val_loss",
                       'model_parameters', 'status', 'error_message', 'config_valid', 'Trial', 'F1_Score',
                       'best_mcc', 'final_mcc', 'best_f1_minority', 'final_f1_minority', 
                       'best_balanced_accuracy', 'final_balanced_accuracy'}
        
        param_cols = [col for col in df.columns if col not in exclude_cols]
        
        for col in param_cols:
            if df[col].dtype in ['object', 'string'] or col.startswith('n_') or 'hidden_dims' in col:
                unique_vals = df[col].unique()
                if len(unique_vals) > 1: 
                    processed_vals = []
                    for val in df[col]:
                        if isinstance(val, str) and ('[' in val or 'True' in val or 'False' in val):
                            processed_vals.append(str(val)[:20] + '...' if len(str(val)) > 20 else str(val))
                        else:
                            processed_vals.append(str(val))
                    
                    unique_processed = list(set(processed_vals))
                    val_map = {val: i for i, val in enumerate(unique_processed)}
                    
                    dimensions.append(dict(
                        label=col,
                        values=[val_map.get(pv, 0) for pv in processed_vals],
                        tickvals=list(range(len(unique_processed))),
                        ticktext=unique_processed
                    ))
            else:
                # Numerical parameter
                if df[col].nunique() > 1:  
                    dimensions.append(dict(
                        label=col,
                        values=df[col],
                        range=[df[col].min(), df[col].max()]
                    ))
        

        fig = go.Figure(data=go.Parcoords(
            line=dict(color=df[eval_metric], colorscale='Viridis', showscale=True,
                     colorbar=dict(title="MCC Score")), 
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title=f"HPO Results: {self.study_name} ({len(df)} completed trials)",
            height=600,
            font=dict(size=10)
        )
        
        return fig
    
    def run(self, n_trials: int = 100, use_wandb: bool = False) -> optuna.Study:
        """Run optimization with experiment-level W&B tracking"""

        if use_wandb:
            self.wandb_run = wandb.init(
                project=self.config.get('wandb_project', 'HPO'),
                group=None,  # This is the parent experiment
                name=self.experiment_id,
                job_type="hpo_experiment", 
                tags=[self.study_name, "hpo_experiment"],
                config={
                    **self.config, 
                    'search_space': self.search_space,
                    'fixed_params': self.fixed_params,
                    'n_trials': n_trials,
                    'experiment_id': self.experiment_id
                }
            )
        
        logger.info(f"Starting HPO Experiment: {self.experiment_id}")
        logger.info(f"Study: {self.study_name} with {n_trials} trials")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"CSV tracking: {self.csv_file}")
        
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=self.n_startup_trials,
            constraints_func=self._optuna_constraints,
        )
        study = optuna.create_study(
            direction="maximize", 
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.pruner_startup,
                n_warmup_steps=self.pruner_warmup,
            ),
            sampler=sampler,
        )
        
        def experiment_callback(study, trial):
            if (trial.state == optuna.trial.TrialState.COMPLETE and 
                self.wandb_run is not None and 
                not self.wandb_run._backend and  
                wandb.run is not None):
                
                try:
                    completed_trials = len([t for t in study.trials 
                                        if t.state == optuna.trial.TrialState.COMPLETE])
                    best_mcc = study.best_value if study.best_value > -1 else -1  
 
                    wandb.log({
                        "experiment/completed_trials": completed_trials,
                        "experiment/best_mcc_so_far": best_mcc,  
                        "experiment/trial_number": trial.number,
                        "experiment/current_trial_mcc": trial.user_attrs.get('best_mcc', -1),  
                    })
                except Exception as e:
                    logger.warning(f"Failed to log to W&B in callback: {e}")
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, callbacks=[experiment_callback])
     
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if self.wandb_run and completed:
            try:
                # Log final experiment summary
                all_mccs = [t.user_attrs.get('best_mcc', -1) for t in completed]  
                wandb.log({
                    "experiment/final_best_mcc": max(all_mccs),  
                    "experiment/mean_mcc": np.mean(all_mccs),  
                    "experiment/std_mcc": np.std(all_mccs), 
                    "experiment/total_completed": len(completed),
                    "experiment/success_rate": len(completed) / len(study.trials)
                })
            except Exception as e:
                logger.warning(f"Failed to log final summary to W&B: {e}")
        self._save_results(study)
    
        
        if self.wandb_run and wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish W&B run: {e}")
    
        return study
    
    def _save_results(self, study: optuna.Study):
        """Save results and create visualizations with robust CSV handling"""
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed:
            logger.error("No completed trials!")
            return
        
        best_mcc = study.best_value if study.best_value > -1 else -1  
        best_trial_from_optuna = study.best_trial.number

        try:
            if self.csv_file.exists() and self.csv_file.stat().st_size > 0:
                try:
                    df = pd.read_csv(self.csv_file, quoting=1)
                except:
                    try:
                        df = pd.read_csv(self.csv_file, quoting=1, on_bad_lines='skip')
                    except:
                        df = pd.read_csv(self.csv_file, on_bad_lines='skip')
                
                completed_df = df[df['status'] == 'completed']
                if len(completed_df) > 0:
                    best_row = completed_df.loc[completed_df['best_mcc'].idxmax()]
                    best_mcc_from_csv = best_row['best_mcc'] 
                    best_trial_num = best_row['trial_number']
                else:
                    best_mcc_from_csv = best_mcc
                    best_trial_num = best_trial_from_optuna
            else:
                best_mcc_from_csv = best_mcc
                best_trial_num = best_trial_from_optuna
        except Exception as e:
            logger.warning(f"Could not read CSV file: {e}. Using Optuna data.")
            best_mcc_from_csv = best_mcc
            best_trial_num = best_trial_from_optuna
        

        best_config = self.config.copy()
        best_config.update(self.fixed_params)
        best_config.update(study.best_params)
        best_config['_hpo_metadata'] = {
            'best_mcc': best_mcc,  
            'best_trial': best_trial_num,
            'total_trials': len(study.trials),
            'completed_trials': len(completed),
            'experiment_id': self.experiment_id,
            'csv_file': str(self.csv_file.name)
        }
        
        config_file = self.results_dir / "best_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(best_config, f)

        fig = self.create_parallel_plot(study)
        if fig:
            plot_file = self.results_dir / "parallel_coordinates.html"
            fig.write_html(plot_file)
            logger.info(f"Interactive plot saved: {plot_file}")

        with open(self.results_dir / "study.pkl", 'wb') as f:
            pickle.dump(study, f)
       
        logger.info(f"\nHPO Experiment Complete!")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Best MCC Score: {best_mcc:.4f} (Trial #{best_trial_num})")  
        logger.info(f"Completed Trials: {len(completed)}/{len(study.trials)}")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info(f"CSV data: {self.csv_file}")
        logger.info(f"Best config: {config_file}")
        if fig:
            logger.info(f"Interactive plot: {plot_file}")

        try:
            if self.csv_file.exists():
                df = pd.read_csv(self.csv_file, quoting=1, on_bad_lines='skip')
                completed_df = df[df['status'] == 'completed']
                if len(completed_df) > 0:
                    top_5 = completed_df.nlargest(5, 'best_mcc')[['trial_number', 'best_mcc', "starting_val_loss", 'best_val_loss', 'final_epoch']]
                    logger.info(f"\nTop 5 Trials from CSV:")
                    logger.info(top_5.to_string(index=False))
                else:
                    raise ValueError("No completed trials in CSV")
            else:
                raise FileNotFoundError("CSV file not found")
        except Exception as e:
            logger.warning(f"Could not show top trials from CSV: {e}")
            logger.info(f"\nTop 5 Trials from Optuna:")
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            top_trials = sorted(completed_trials, key=lambda t: t.user_attrs.get('best_mcc', t.value), reverse=True)[:5]
            for i, trial in enumerate(top_trials):
                mcc = trial.user_attrs.get('best_mcc', trial.value)  
                val_loss = trial.user_attrs.get('best_val_loss', 'N/A')
                epoch = trial.user_attrs.get('final_epoch', 'N/A')
                logger.info(f"  Trial {trial.number}: MCC={mcc:.4f}, Val_Loss={val_loss}, Epoch={epoch}")  


def main():
    parser = argparse.ArgumentParser(description="Simplified HPO")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--study_name", type=str, default="hpo", help="Study name")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--wandb", action="store_true", help="Use W&B")
    
    args = parser.parse_args()
    
    config_path = Path("model_search/config_files") / args.config
    hpo = SimpleHPO(config_path, args.study_name)
    hpo.run(args.trials, args.wandb)


if __name__ == "__main__":
    main()