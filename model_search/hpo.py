import argparse
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


class SimpleHPO:
    """Simplified HPO with feasibility checks, CSV tracking, and W&B experiment grouping"""
    
    def __init__(self, config_path: str, study_name: str, startup_trials: int = 10,
                  pruner_startup: int = 5, pruner_warmup: int = 10):
        self.n_startup_trials = startup_trials
        self.pruner_startup = pruner_startup
        self.pruner_warmup = pruner_warmup
        self.seed = 0 
        self.config_path = config_path
        self.study_name = study_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(config_path, 'r') as f:
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
            df.to_csv(self.csv_file, index=False, quoting=1)  # Quote all fields
        else:
            df.to_csv(self.csv_file, mode='a', header=False, index=False, quoting=1)
    
    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Suggest parameters with FIXED feasibility checks using YAML config"""
        config = self.config.copy()
        config.update(self.fixed_params)
        
        # Phase 1: Suggest GAT heads FIRST (this affects all dimension constraints)
        if 'gat_heads' not in config and 'gat_heads' in self.search_space:
            spec = self.search_space['gat_heads']
            if spec['search_type'] == 'categorical':
                config['gat_heads'] = trial.suggest_categorical('gat_heads', spec['choices'])
            elif spec['search_type'] == 'int':
                config['gat_heads'] = trial.suggest_int('gat_heads', spec['min'], spec['max'])
        
        # Phase 2: Suggest GNN hidden dim with GAT constraint awareness
        if 'gnn_hidden_dim' not in config and 'gnn_hidden_dim' in self.search_space:
            spec = self.search_space['gnn_hidden_dim']
            gat_heads = config.get('gat_heads', 1)  # Default to 1 if not GAT
            
            if spec['search_type'] == 'categorical':
                # Filter choices to be divisible by GAT heads
                valid_choices = [dim for dim in spec['choices'] if dim % gat_heads == 0]
                if not valid_choices:
                    # Fallback: find nearest multiples
                    valid_choices = [((dim // gat_heads) + 1) * gat_heads for dim in spec['choices']]
                    valid_choices = list(set(valid_choices))  # Remove duplicates
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
                        valid_choices = [1]  # Safe fallback
                config['switch_attention_heads'] = trial.suggest_categorical('switch_attention_heads', valid_choices)
            elif spec['search_type'] == 'int':
                # Find valid divisors within range
                valid_heads = []
                for heads in range(spec['min'], spec['max'] + 1):
                    if gnn_hidden_dim % heads == 0:
                        valid_heads.append(heads)
                if not valid_heads:
                    valid_heads = [1]  # Safe fallback
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
        """Handle dynamic list parameters like hidden_dims"""
        n_layers = trial.suggest_int(f'n_{param}_layers', 
                                   spec.get('n_layers_min', 1), 
                                   spec.get('n_layers_max', 3))
        
        dims = []
        for i in range(n_layers):
            dim = trial.suggest_int(f'{param}_{i}', 
                                  spec.get('dim_min', 32), 
                                  spec.get('dim_max', 256), 
                                  step=spec.get('dim_step', 32))
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
                cons.append(float(gnn_hidden_dim % gat_heads))
        
        # Constraint 2: Switch attention heads divisibility
        switch_head_type = p.get('switch_head_type', '')
        if 'attention' in switch_head_type.lower():
            gnn_hidden_dim = p.get('gnn_hidden_dim')
            switch_heads = p.get('switch_attention_heads')
            if gnn_hidden_dim is not None and switch_heads is not None:
                cons.append(float(gnn_hidden_dim % switch_heads))
                #cons.append(float(p['hidden_dim'] % p['switch_attention_heads']))
        
        # Constraint 3: At least one MLP must be enabled
        use_node_mlp = p.get('use_node_mlp', True)
        use_edge_mlp = p.get('use_edge_mlp', True)
        if not use_node_mlp and not use_edge_mlp:
            cons.append(1.0)  # Violation
        else:
            cons.append(0.0)  # Satisfied
        
        # Constraint 4: PhyR requires k_ratio
        use_phyr = p.get('use_phyr', False)
        phyr_k_ratio = p.get('phyr_k_ratio')
        if use_phyr and phyr_k_ratio is None:
            cons.append(1.0)  # Violation
        else:
            cons.append(0.0)  # Satisfied
        
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
        output_type = config.get('output_type', 'binary')
        if output_type == 'multiclass' and 'criterion_name' not in self.fixed_params:
            config['criterion_name'] = 'CrossEntropyLoss'
        elif output_type == 'binary' and 'criterion_name' not in self.fixed_params:
            config['criterion_name'] = 'BCEWithLogitsLoss'
        elif output_type == 'regression' and 'criterion_name' not in self.fixed_params:
            config['criterion_name'] = 'MSELoss'
        
        # Constraint 13: Set num_classes based on output_type
        if output_type == 'multiclass':
            config['num_classes'] = 2  # Fixed to 2 classes as requested
        elif output_type == 'binary':
            config['num_classes'] = 2
        else:  # regression
            config.pop('num_classes', None)
        
        # Constraint 14: Memory limit (batch_size × max_nodes)
        batch_size = config.get('batch_size', 32)
        max_nodes = config.get('max_nodes', 1000)
        memory_limit = 100000  # Adjust based on your GPU memory
        
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
        """Simplified objective function with comprehensive constraint validation"""
        
        # Initialize individual trial W&B run
        trial_run = None
        if self.wandb_run:
            trial_run = wandb.init(
                project=self.config.get('wandb_project', 'HPO'),
                group=self.experiment_id,  # Group all trials together
                name=f"trial_{trial.number:03d}",
                job_type="hpo_trial",
                tags=[self.study_name, "hpo_trial"],
                config=None,  # Will set later
                reinit=True
            )
        
        try:
            config = self.suggest_params(trial)
            
            # Validate configuration
            if not self._validate_config(config):
                logger.warning(f"Trial {trial.number}: Invalid configuration, pruning")
                if trial_run:
                    wandb.log({"status": "invalid_config"})
                    wandb.finish()
                raise optuna.TrialPruned()
            
            # Set W&B config for this trial
            if trial_run:
                wandb.config.update(config)
            
            # Create model and data
            model, train_loader, val_loader, criterion = self._setup_training(config)
            if model is None:
                if trial_run:
                    wandb.log({"status": "failed_setup"})
                    wandb.finish()
                return 1.0
            
            # Log configuration validity and model size
            model_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Trial {trial.number}: Config valid, Model params: {model_params:,}")
            
            # Training loop with detailed logging
            optimizer = optim.Adam(model.parameters(), 
                                 lr=config['learning_rate'], 
                                 weight_decay=config['weight_decay'])
            
            best_f1 = 0.0
            best_train_loss = float('inf')
            best_val_loss = float('inf')
            starting_val_loss = float('inf')
            final_train_loss = float('inf')
            final_val_loss = float('inf')
            final_precision = 0.0
            final_recall = 0.0
            final_accuracy = 0.0
            
            
            patience = 0
            max_epochs = min(config.get('epochs', 100), 80)
            max_patience = config.get('patience', 25)
            
            for epoch in range(max_epochs):
                train_loss, train_dict = train(model, train_loader, optimizer, criterion, self.device)
                val_loss, val_dict = test(model, val_loader, criterion, self.device)
                
                if epoch == 0: 
                    starting_val_loss = val_loss
                # Extract metrics
                current_f1 = val_dict.get('test_f1', 0.0)
                current_precision = val_dict.get('test_precision', 0.0)
                current_recall = val_dict.get('test_recall', 0.0)
                current_accuracy = val_dict.get('test_accuracy', 0.0)
                
                # Fallback F1 calculation if not available
                if current_f1 == 0.0 and current_precision + current_recall > 0:
                    current_f1 = 2 * current_precision * current_recall / (current_precision + current_recall + 1e-8)
                
                # Track best and final metrics
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_train_loss = train_loss
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1
                
                # Always update final metrics
                final_train_loss = train_loss
                final_val_loss = val_loss
                final_precision = current_precision
                final_recall = current_recall
                final_accuracy = current_accuracy
                
                # Log to W&B every epoch
                if trial_run:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "f1_score": current_f1,
                        "precision": current_precision,
                        "recall": current_recall,
                        "accuracy": current_accuracy,
                        "best_f1": best_f1,
                        "patience": patience
                    })
                
                # Pruning and early stopping
                trial.report(1 - current_f1, epoch)
                if trial.should_prune() or patience >= max_patience:
                    break
            
            # Comprehensive metrics for CSV and Optuna
            metrics = {
                "starting_val_loss": starting_val_loss,
                'best_f1_score': best_f1,
                'best_train_loss': best_train_loss,
                'best_val_loss': best_val_loss,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'final_precision': final_precision,
                'final_recall': final_recall,
                'final_accuracy': final_accuracy,
                'final_epoch': epoch,
                'converged': patience < max_patience,
                'model_parameters': model_params,
                'status': 'completed',
                'config_valid': True
            }
            
            # Store in Optuna trial
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            
            # Log final metrics to W&B
            if trial_run:
                wandb.log(metrics)
                wandb.log({"trial_score": 1 - best_f1})  # The objective value
                wandb.finish()
            
            # Log to CSV
            self._log_trial_to_csv(trial.number, config, metrics)
            
            logger.info(f"Trial {trial.number}: F1={best_f1:.4f}, Val_Loss={best_val_loss:.4f}, Epoch={epoch}, Params={model_params:,}")
            
            return 1 - best_f1  # Minimize
            
        except optuna.TrialPruned:
            # Handle pruned trials properly
            pruned_metrics = {
                "starting_val_loss": float('inf'),
                'best_f1_score': 0.0,
                'best_train_loss': float('inf'),
                'best_val_loss': float('inf'),
                'final_train_loss': float('inf'),
                'final_val_loss': float('inf'),
                'final_precision': 0.0,
                'final_recall': 0.0,
                'final_accuracy': 0.0,
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
            
            # Log failure
            failure_metrics = {
                "starting_val_loss": float('inf'),
                'best_f1_score': 0.0,
                'best_train_loss': float('inf'),
                'best_val_loss': float('inf'),
                'final_train_loss': float('inf'),
                'final_val_loss': float('inf'),
                'final_precision': 0.0,
                'final_recall': 0.0,
                'final_accuracy': 0.0,
                'final_epoch': 0,
                'converged': False,
                'model_parameters': 0,
                'status': 'failed',
                'config_valid': False,
                'error_message': str(e)
            }
            
            if trial_run:
                wandb.log(failure_metrics)
                wandb.finish()
            
            self._log_trial_to_csv(trial.number, self.suggest_params(trial), failure_metrics)
            
            return 1.0
    def _load_data_once(self):
        """Load data once during initialization to avoid repeated loading"""
        try:
            logger.info("Loading data once for all trials...")
            
            # Create data loaders using config
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
            
            # Store a sample for getting dimensions
            self.data_sample = self.train_loader.dataset[0]
            
            logger.info(f"Data loaded successfully: {len(self.train_loader)} train batches, {len(self.val_loader)} val batches")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    def _create_dynamic_data_loader(self, batch_size: int):
        """Create new data loaders with different batch size if needed"""
        if batch_size == self.config['batch_size']:
            # Use pre-loaded loaders
            return self.train_loader, self.val_loader
        
        # Create new loaders with different batch size
        dataloaders = create_data_loaders(
            dataset_names=self.config['dataset_names'],
            folder_names=self.config['folder_names'],
            dataset_type=self.config.get('dataset_type', 'default'),
            batch_size=batch_size,  # Use trial-specific batch size
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
            
            # Get data loaders (either pre-loaded or create new ones for different batch size)
            train_loader, val_loader = self._create_dynamic_data_loader(config['batch_size'])
            
            if not train_loader or not val_loader:
                return None, None, None, None
            
            # Get dimensions from pre-loaded sample
            model_kwargs = {
                'node_input_dim': self.data_sample.x.shape[1],
                'edge_input_dim': self.data_sample.edge_attr.shape[1],
            }
            
            # Add relevant parameters
            for key in ['activation', 'dropout_rate', 'gnn_type', 'gnn_layers', 'gnn_hidden_dim', 
                       'gat_heads', 'gat_dropout', 'gin_eps', 'use_node_mlp', 'use_edge_mlp',
                       'node_hidden_dims', 'edge_hidden_dims', 'use_batch_norm', 'use_residual',
                       'use_skip_connections', 'switch_head_type', 'switch_head_layers', 
                       'switch_attention_heads', 'output_type', 'num_classes', 'use_gated_mp',
                       'use_phyr', 'phyr_k_ratio', 'pooling', 'normalization_type', 
                       'loss_scaling_strategy']:
                if key in config:
                    model_kwargs[key] = config[key]
            
            model = model_class(**model_kwargs).to(self.device)
            
            # Create criterion
            criterion_name = config.get('criterion_name', 'MSELoss')
            if hasattr(model_module, criterion_name):
                criterion = getattr(model_module, criterion_name)()
            else:
                criterion = getattr(nn, criterion_name)()
            
            return model, train_loader, val_loader, criterion
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return None, None, None, None
    
    def create_parallel_plot(self, study: optuna.Study) -> go.Figure:
        """Create parallel coordinates plot with robust CSV handling"""
        # Try to read from CSV first, fallback to Optuna data
        try:
            if self.csv_file.exists() and self.csv_file.stat().st_size > 0:
                try:
                    df = pd.read_csv(self.csv_file, quoting=1, on_bad_lines='skip')
                    df = df[df['status'] == 'completed'].sort_values('best_f1_score', ascending=False)
                    
                    if len(df) >= 5:
                        # Use CSV data
                        return self._create_plot_from_dataframe(df)
                except Exception as e:
                    logger.warning(f"Could not read CSV for plotting: {e}")
        except Exception:
            pass
        
        # Fallback to Optuna data
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) < 5:
            return None
        
        # Build data from Optuna trials
        data = []
        for trial in completed_trials:
            row = trial.params.copy()
            # Handle list parameters properly
            for key, value in row.items():
                if isinstance(value, list):
                    row[key] = str(value)
            row['F1_Score'] = trial.user_attrs.get('best_f1_score', 1 - trial.value)
            row['Trial'] = trial.number
            data.append(row)
        
        df = pd.DataFrame(data).sort_values('F1_Score', ascending=False)
        return self._create_plot_from_dataframe(df, f1_col='F1_Score')
    
    def _create_plot_from_dataframe(self, df: pd.DataFrame, f1_col: str = 'best_f1_score') -> go.Figure:
        """Create parallel coordinates plot from DataFrame"""
        # Create dimensions - start with F1 score
        dimensions = [
            dict(label="F1 Score", values=df[f1_col], 
                 range=[df[f1_col].min(), df[f1_col].max()])
        ]
        
        # Add other metrics if available
        for metric in ['best_val_loss', 'final_epoch']:
            if metric in df.columns and df[metric].notna().any():
                dimensions.append(dict(
                    label=metric.replace('_', ' ').title(),
                    values=df[metric],
                    range=[df[metric].min(), df[metric].max()]
                ))
        
        # Add hyperparameters (exclude metadata columns)
        exclude_cols = {'trial_number', 'timestamp', 'best_f1_score', 'best_train_loss', 
                       'best_val_loss', 'final_train_loss', 'final_val_loss', 'final_precision',
                       'final_recall', 'final_accuracy', 'final_epoch', 'converged', "starting_val_loss",
                       'model_parameters', 'status', 'error_message', 'config_valid', 'Trial', 'F1_Score'}
        
        param_cols = [col for col in df.columns if col not in exclude_cols]
        
        for col in param_cols:
            if df[col].dtype in ['object', 'string'] or col.startswith('n_') or 'hidden_dims' in col:
                # Categorical or list parameters
                unique_vals = df[col].unique()
                if len(unique_vals) > 1:  # Only include if there's variation
                    # Handle string representations of lists
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
                if df[col].nunique() > 1:  # Only include if there's variation
                    dimensions.append(dict(
                        label=col,
                        values=df[col],
                        range=[df[col].min(), df[col].max()]
                    ))
        
        # Create plot
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=df[f1_col], colorscale='Viridis', showscale=True,
                     colorbar=dict(title="F1 Score")),
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
        
        # Initialize experiment-level W&B run
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
        
        # Create study
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=self.n_startup_trials,
            constraints_func=self._optuna_constraints,
        )
        study = optuna.create_study(
            direction= "minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.pruner_startup,
                n_warmup_steps=self.pruner_warmup,
            ),
            sampler=sampler,
        )
        
        # Add callback for experiment-level logging
        def experiment_callback(study, trial):
            # Only log if we have an active W&B run and trial completed successfully
            if (trial.state == optuna.trial.TrialState.COMPLETE and 
                self.wandb_run is not None and 
                not self.wandb_run._backend and  
                wandb.run is not None):
                
                try:
                    completed_trials = len([t for t in study.trials 
                                        if t.state == optuna.trial.TrialState.COMPLETE])
                    best_f1 = study.best_value if study.best_value < 1 else 0
                    
                    # Log experiment-level metrics
                    wandb.log({
                        "experiment/completed_trials": completed_trials,
                        "experiment/best_f1_so_far": best_f1,
                        "experiment/trial_number": trial.number,
                        "experiment/current_trial_f1": trial.user_attrs.get('best_f1_score', 0),
                    })
                except Exception as e:
                    logger.warning(f"Failed to log to W&B in callback: {e}")
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, callbacks=[experiment_callback])
        
        # Final experiment summary
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if self.wandb_run and completed:
            try:
                # Log final experiment summary
                all_f1s = [t.user_attrs.get('best_f1_score', 0) for t in completed]
                wandb.log({
                    "experiment/final_best_f1": max(all_f1s),
                    "experiment/mean_f1": np.mean(all_f1s),
                    "experiment/std_f1": np.std(all_f1s),
                    "experiment/total_completed": len(completed),
                    "experiment/success_rate": len(completed) / len(study.trials)
                })
            except Exception as e:
                logger.warning(f"Failed to log final summary to W&B: {e}")
    
        
        # Save results
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
        
        # Best results - use Optuna data as fallback if CSV fails
        best_f1_from_optuna = 1 - study.best_value if study.best_value < 1 else 0
        best_trial_from_optuna = study.best_trial.number
        
        # Try to read CSV, fallback to Optuna data if it fails
        try:
            if self.csv_file.exists() and self.csv_file.stat().st_size > 0:
                # Try reading with different options to handle parsing errors
                try:
                    df = pd.read_csv(self.csv_file, quoting=1)
                except:
                    # If that fails, try with different settings
                    try:
                        df = pd.read_csv(self.csv_file, quoting=1, on_bad_lines='skip')
                    except:
                        # Last resort: read without quotes
                        df = pd.read_csv(self.csv_file, on_bad_lines='skip')
                
                completed_df = df[df['status'] == 'completed']
                if len(completed_df) > 0:
                    best_row = completed_df.loc[completed_df['best_f1_score'].idxmax()]
                    best_f1 = best_row['best_f1_score']
                    best_trial_num = best_row['trial_number']
                else:
                    best_f1 = best_f1_from_optuna
                    best_trial_num = best_trial_from_optuna
            else:
                best_f1 = best_f1_from_optuna
                best_trial_num = best_trial_from_optuna
        except Exception as e:
            logger.warning(f"Could not read CSV file: {e}. Using Optuna data.")
            best_f1 = best_f1_from_optuna
            best_trial_num = best_trial_from_optuna
        
        # Save best config
        best_config = self.config.copy()
        best_config.update(self.fixed_params)
        best_config.update(study.best_params)
        best_config['_hpo_metadata'] = {
            'best_f1_score': best_f1,
            'best_trial': best_trial_num,
            'total_trials': len(study.trials),
            'completed_trials': len(completed),
            'experiment_id': self.experiment_id,
            'csv_file': str(self.csv_file.name)
        }
        
        config_file = self.results_dir / "best_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(best_config, f)
        
        # Save parallel coordinates plot  
        fig = self.create_parallel_plot(study)
        if fig:
            plot_file = self.results_dir / "parallel_coordinates.html"
            fig.write_html(plot_file)
            logger.info(f"Interactive plot saved: {plot_file}")
        
        # Save study
        with open(self.results_dir / "study.pkl", 'wb') as f:
            pickle.dump(study, f)
        
        # Print summary
        logger.info(f"\nHPO Experiment Complete!")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Best F1 Score: {best_f1:.4f} (Trial #{best_trial_num})")
        logger.info(f"Completed Trials: {len(completed)}/{len(study.trials)}")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info(f"CSV data: {self.csv_file}")
        logger.info(f"Best config: {config_file}")
        if fig:
            logger.info(f"Interactive plot: {plot_file}")
        
        # Show top 5 results from Optuna if CSV fails
        try:
            if self.csv_file.exists():
                df = pd.read_csv(self.csv_file, quoting=1, on_bad_lines='skip')
                completed_df = df[df['status'] == 'completed']
                if len(completed_df) > 0:
                    top_5 = completed_df.nlargest(5, 'best_f1_score')[['trial_number', 'best_f1_score',"starting_val_loss", 'best_val_loss', 'final_epoch']]
                    logger.info(f"\nTop 5 Trials from CSV:")
                    logger.info(top_5.to_string(index=False))
                else:
                    raise ValueError("No completed trials in CSV")
            else:
                raise FileNotFoundError("CSV file not found")
        except Exception as e:
            logger.warning(f"Could not show top trials from CSV: {e}")
            # Show top 5 from Optuna data instead
            logger.info(f"\nTop 5 Trials from Optuna:")
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            top_trials = sorted(completed_trials, key=lambda t: t.user_attrs.get('best_f1_score', 1-t.value), reverse=True)[:5]
            for i, trial in enumerate(top_trials):
                f1 = trial.user_attrs.get('best_f1_score', 1-trial.value)
                val_loss = trial.user_attrs.get('best_val_loss', 'N/A')
                epoch = trial.user_attrs.get('final_epoch', 'N/A')
                logger.info(f"  Trial {trial.number}: F1={f1:.4f}, Val_Loss={val_loss}, Epoch={epoch}")


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