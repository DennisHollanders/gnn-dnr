import argparse
from re import A
from typing import final, Dict, Any
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import optuna
import random
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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import threading
import time
from functools import partial
import json
import signal
from contextlib import contextmanager

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


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("HPO")

# Add paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "model_search")])
from model_search.models.AdvancedMLP.AdvancedMLP import AdvancedMLP
from load_data import create_data_loaders
from train import train, test
# Add paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "src")])

from loss_functions import FocalLoss, WeightedBCELoss

WORKER_DATALOADERS = {}

def set_global_determinism(seed: int):
    """
    Sets seeds for all relevant libraries and configures PyTorch for deterministic operations.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enforce deterministic algorithms in PyTorch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
class HPO:
    """Parallelized HPO with multi-process execution and async TPE updates"""
    
    def __init__(self, config_path: str, study_name: str, startup_trials: int = 25,
                 pruner_startup: int = 15, pruner_warmup: int = 15, n_parallel: int = None,
                 trial_timeout: int = 3600, stage1_checkpoint: str=None):  # Add trial timeout parameter (default 1 hour)
        self.n_startup_trials = startup_trials
        self.pruner_startup = pruner_startup
        self.pruner_warmup = pruner_warmup


        self.stage1_checkpoint = stage1_checkpoint
        self.config_path = config_path
        self.study_name = study_name
        self.trial_timeout = trial_timeout  
        
        # Force CPU usage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cpu':
            torch.set_num_threads(1)
        torch.set_num_threads(1)  # Prevent thread oversubscription
        
        if n_parallel is None:
            n_cpus = mp.cpu_count()
            self.n_parallel = max(2, min(n_cpus // 2, 8))
        else:
            self.n_parallel = n_parallel
            
        logger.info(f"Using {self.n_parallel} parallel workers on CPU")
        logger.info(f"Trial timeout set to {self.trial_timeout} seconds")
        
        with open(config_path, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.seed = self.config.get('seed', 0) 
        
        
        self.search_space = self.config.pop('search_space', {})
        self.fixed_params = self.config.pop('fixed_params', {})
        
        # Results directory
        model_name = self.config.get('model_module', 'Unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path(f"model_search/models/{model_name}/hpo_{study_name}_{timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV tracking with thread safety
        self.csv_file = self.results_dir / "hpo_results.csv"
        self._csv_headers = ['trial_number', 'state', 'value', 'datetime_start', 'datetime_complete']
        self._csv_headers += sorted(self.search_space.keys())
        self._csv_headers += ['best_mcc', 'final_val_loss', 'final_epoch', 'model_parameters', 'elapsed_time']
        self._csv_headers = list(dict.fromkeys(self._csv_headers))
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_headers)
            writer.writeheader()
        self.csv_lock = threading.Lock()

        self.wandb_run = None
        self.experiment_id = f"{study_name}_{timestamp}"

        # Data loading strategy - load once per worker
        self.train_loader = None
        self.val_loader = None
        self.data_sample = None
    
    def _init_csv_tracking(self):
        """Initialize CSV file for tracking all trial results"""
        # Create CSV with headers for hyperparameters + metrics
        self.csv_file.touch()
        
    def _log_trial_to_csv_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):

        row_data = {
            'trial_number': trial.number,
            'state': trial.state.name,
            'value': trial.value,
            'datetime_start': trial.datetime_start,
            'datetime_complete': trial.datetime_complete,
        }

        # Explicitly update with all suggested parameters
        row_data.update(trial.params)

        # Add user-set metrics if they exist
        if "metrics" in trial.user_attrs:
            row_data.update(trial.user_attrs["metrics"])

        final_row = {header: row_data.get(header) for header in self._csv_headers}

        with self.csv_lock:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_headers)
                writer.writerow(final_row)
    
    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Suggests hyperparameters using a 'suggest then prune' strategy for dependencies."""
        config = self.config.copy()
        config.update(self.fixed_params)

        # Suggest gnn_hidden_dim first, as it's a dependency for attention heads.
        gnn_hidden_dim = None
        if 'gnn_hidden_dim' in self.search_space:
            spec = self.search_space['gnn_hidden_dim']
            gnn_hidden_dim = trial.suggest_categorical('gnn_hidden_dim', spec['choices'])
            config['gnn_hidden_dim'] = gnn_hidden_dim
        else:
            # Fallback to fixed params or a default value
            gnn_hidden_dim = config.get('gnn_hidden_dim', 256)

        # --- GNN Specific Parameters ---
        if config.get('gnn_type') == 'GAT':
            # Suggest GAT heads from the static list of choices
            if 'gat_heads' in self.search_space:
                spec = self.search_space['gat_heads']
                gat_heads = trial.suggest_categorical('gat_heads', spec['choices'])

                # Prune the trial if the combination of parameters is invalid
                if gnn_hidden_dim % gat_heads != 0:
                    raise optuna.exceptions.TrialPruned(
                        f"GAT heads ({gat_heads}) must be a divisor of gnn_hidden_dim ({gnn_hidden_dim})."
                    )
                config['gat_heads'] = gat_heads

            # Suggest other GAT-specific parameters
            for param in ['gat_dropout', 'gat_v2', 'gat_edge_dim']:
                if param in self.search_space:
                    spec = self.search_space[param]
                    if spec.get('search_type') == 'float':
                        config[param] = trial.suggest_float(param, spec['min'], spec['max'], log=spec.get('log', False))
                    elif spec.get('search_type') == 'categorical':
                        config[param] = trial.suggest_categorical(param, spec['choices'])
                    elif spec.get('search_type') == 'int':
                        config[param] = trial.suggest_int(param, spec['min'], spec['max'])
        else:
            # Set default values for GAT if not used
            config['gat_heads'] = 0
            config['gat_dropout'] = 0.0

        if config.get('gnn_type') == 'GIN':
            # Suggest GIN-specific parameters
            for param in ['gin_layers', 'gin_hidden_dim', 'gin_eps']:
                if param in self.search_space:
                    spec = self.search_space[param]
                    if spec.get('search_type') == 'float':
                        config[param] = trial.suggest_float(param, spec['min'], spec['max'], log=spec.get('log', False))
                    elif spec.get('search_type') == 'int':
                        config[param] = trial.suggest_int(param, spec['min'], spec['max'])
        else:
            # Set default values for GIN if not used
            config['gin_layers'] = 0
            config['gin_hidden_dim'] = 0
            config['gin_eps'] = 0.0
            
        # --- Switch Head Parameters ---
        if 'switch_head_type' in self.search_space:
            config['switch_head_type'] = trial.suggest_categorical(
                'switch_head_type', self.search_space['switch_head_type']['choices']
            )
        
        if 'attention' in config.get('switch_head_type', ''):
            # Suggest switch attention heads from the static list of choices
            if 'switch_attention_heads' in self.search_space:
                spec = self.search_space['switch_attention_heads']
                switch_heads = trial.suggest_categorical('switch_attention_heads', spec['choices'])

                # Prune the trial if the combination is invalid
                if gnn_hidden_dim % switch_heads != 0:
                    raise optuna.exceptions.TrialPruned(
                        f"Switch heads ({switch_heads}) must be a divisor of gnn_hidden_dim ({gnn_hidden_dim})."
                    )
                config['switch_attention_heads'] = switch_heads
        else:
            config['switch_attention_heads'] = 0
        
        if 'criterion_name' in self.search_space:
            crit_spec = self.search_space['criterion_name']
            config['criterion_name'] = trial.suggest_categorical(
                'criterion_name', crit_spec['choices']
            )

        # --- Suggest all other parameters from search_space ---
        handled_params = {
            'gnn_hidden_dim', 'gat_heads', 'gat_dropout', 'gat_v2', 'gat_edge_dim',
            'gin_layers', 'gin_hidden_dim', 'gin_eps', 'switch_head_type', 
            'switch_attention_heads'
        }
        mlp_prefixes = ('node_hidden_dim_', 'edge_hidden_dim_')
        
        for param, spec in self.search_space.items():
            if param in config or param in handled_params or param.startswith(mlp_prefixes):
                continue

            search_type = spec.get('search_type')
            if search_type == 'categorical':
                config[param] = trial.suggest_categorical(param, spec['choices'])
            elif search_type == 'int':
                config[param] = trial.suggest_int(param, spec['min'], spec['max'])
            elif search_type == 'float':
                config[param] = trial.suggest_float(param, spec['min'], spec['max'], log=spec.get('log', False))

        # --- MLP Architecture ---
        def suggest_mlp_layers(trial, prefix):
            layer_params = sorted([
                key for key in self.search_space.keys() if key.startswith(f'{prefix}_dim_')
            ])
            dims = []
            for param_name in layer_params:
                if param_name in self.search_space:
                    spec = self.search_space[param_name]
                    dim = trial.suggest_categorical(param_name, spec['choices'])
                    dims.append(dim)
            
            while dims and dims[-1] == 0:
                dims.pop()
            
            return [d for d in dims if d > 0]

        config['node_hidden_dims'] = suggest_mlp_layers(trial, 'node_hidden')
        config['edge_hidden_dims'] = suggest_mlp_layers(trial, 'edge_hidden')

        # Apply final constraints and return
        return self._apply_constraints(trial, config)
    def _apply_constraints(self, trial: optuna.Trial, config: dict) -> dict:
        """Apply comprehensive feasibility constraints with standardized null values"""
        

        
        # Constraint 2: hidden_dim % gat_heads == 0 (if hidden_dim exists)
        if config.get('gnn_type') == 'GAT' and 'hidden_dim' in config:
            hidden_dim = config['hidden_dim']
            gat_heads = config.get('gat_heads', 4)
            
            remainder = hidden_dim % gat_heads
            if remainder != 0:
                config['hidden_dim'] = hidden_dim + (gat_heads - remainder)
        
        # Constraint 3: switch_attention_heads divisibility with gnn_hidden_dim
        if 'attention' in config.get('switch_head_type', '').lower() and 'switch_attention_heads' in config:
            gnn_hidden_dim = config.get('gnn_hidden_dim', 128)
            switch_heads = config.get('switch_attention_heads', 1)
            
            if switch_heads > 0 and gnn_hidden_dim % switch_heads != 0:
                # Adjust gnn_hidden_dim to be divisible by switch_heads
                remainder = gnn_hidden_dim % switch_heads
                config['gnn_hidden_dim'] = gnn_hidden_dim + (switch_heads - remainder)
                logger.debug(f"Adjusted gnn_hidden_dim to {config['gnn_hidden_dim']} for switch attention heads")
        
        # Constraint 4: Conditional parameters for GAT
        if config.get('gnn_type') != 'GAT':
            # Set GAT-specific parameters to default values instead of popping
            config['gat_heads'] = 0
            config['gat_dropout'] = 0.0
            config['gat_v2'] = False
            config['gat_edge_dim'] = 0
        
        # Constraint 5: Conditional parameters for GIN
        if config.get('gnn_type') != 'GIN':
            # Set GIN-specific parameters to default values
            config['gin_eps'] = 0.0
            config['gin_train_eps'] = False
            config['gin_mlp_layers'] = 0
        
        # Constraint 6: Switch attention heads only for attention-based switch heads
        switch_head_type = config.get('switch_head_type', 'mlp')
        if 'attention' not in switch_head_type.lower():
            config['switch_attention_heads'] = 0
        elif 'switch_attention_heads' not in config:
            # Set a safe default if using attention but no heads specified
            config['switch_attention_heads'] = 1
        
        # Constraint 7: Node MLP dims only if using node MLP
        if not config.get('use_node_mlp', True):
            config['node_hidden_dims'] = []  # Empty list instead of removing
            
        # Constraint 8: Edge MLP dims only if using edge MLP  
        if not config.get('use_edge_mlp', True):
            config['edge_hidden_dims'] = []  # Empty list instead of removing
        
        # Constraint 9: Ensure at least one MLP is used
        if not config.get('use_node_mlp') and not config.get('use_edge_mlp'):
            config['use_node_mlp'] = True
            if 'node_hidden_dims' not in config or not config['node_hidden_dims']:
                config['node_hidden_dims'] = [128]
        
        # Constraint 10: Gated MP requires GNN
        if config.get('use_gated_mp') and not config.get('gnn_type'):
            config['use_gated_mp'] = False
            logger.debug("Disabled gated MP because no GNN type specified")
        
        # Constraint 11: PhyR parameters only if using PhyR
        if not config.get('use_phyr', False):
            config['phyr_k_ratio'] = 0.0  # Default value instead of removing
        
        # Constraint 12: Match criterion to output type
        output_type = config.get('output_type', 'multiclass')
        if output_type == 'multiclass' and 'criterion_name' not in self.fixed_params:
            config['criterion_name'] = "WeightedBCELoss"
        elif output_type == 'binary' and 'criterion_name' not in self.fixed_params:
            config['criterion_name'] = 'BCEWithLogitsLoss'
        elif output_type == 'regression' and 'criterion_name' not in self.fixed_params:
            config['criterion_name'] = 'MSELoss'
        
        # Constraint 13: Set GNN-specific params to defaults if no GNN
        if not config.get('gnn_type') or config.get('gnn_type') == 'NONE':
            config['gnn_layers'] = 0
            config['gnn_hidden_dim'] = 0
            # Set all GNN-specific params to defaults
            config['gat_heads'] = 0
            config['gat_dropout'] = 0.0
            config['gin_eps'] = 0.0
            config['gin_train_eps'] = False
            config['gin_mlp_layers'] = 0
        
        # Constraint 14: Memory limit (batch_size × max_nodes)
        batch_size = config.get('batch_size', 32)
        max_nodes = config.get('max_nodes', 1000)
        memory_limit = 100000  
        
        if batch_size * max_nodes > memory_limit:
            # Reduce batch size to fit memory
            config['batch_size'] = max(1, memory_limit // max_nodes)
            logger.debug(f"Reduced batch_size to {config['batch_size']} for memory constraint")
        
        # Ensure all parameters have consistent types
        # This prevents mixed types in CSV logging
        for param, expected_type in {
            'use_node_mlp': bool,
            'use_edge_mlp': bool,
            'gin_train_eps': bool,
            'use_phyr': bool,
            'enforce_radiality': bool,
            'use_gated_mp': bool,
            'gnn_layers': int,
            'switch_head_layers': int,
            'gin_mlp_layers': int,
            'gat_heads': int,
            'switch_attention_heads': int,
        }.items():
            if param in config and not isinstance(config[param], expected_type):
                if expected_type == bool:
                    config[param] = bool(config[param])
                elif expected_type == int:
                    config[param] = int(config[param])
        
        
        return config
    
    def _validate_config(self, config: dict) -> bool:
        """Validate configuration before trial execution"""

        # GAT params: only invalid if non‐GAT backbone AND non‐zero GAT settings
        if config.get('gnn_type') != 'GAT':
            gat_params = ['gat_heads', 'gat_dropout', 'gat_v2', 'gat_edge_dim']
            # if any of these is not the default zero/False, it's truly invalid
            if any(config.get(p) not in (0, 0.0, False, None) for p in gat_params):
                logger.warning("Invalid config: non‐zero GAT parameters with non‐GAT backbone")
                logger.warning(f"config with problem: {config}")
                return False

        # GIN params: same logic
        if config.get('gnn_type') != 'GIN':
            gin_params = ['gin_eps', 'gin_train_eps', 'gin_mlp_layers']
            if any(config.get(p) not in (0, 0.0, False, None) for p in gin_params):
                logger.warning("Invalid config: non‐zero GIN parameters with non‐GIN backbone")
                logger.warning(f"config with problem: {config}")
                return False

        # Check divisibility constraints for real GAT configs
        if config.get('gnn_type') == 'GAT':
            if config['gnn_hidden_dim'] % config['gat_heads'] != 0:
                logger.warning(f"Invalid GAT config: {config['gnn_hidden_dim']} not divisible by {config['gat_heads']}")
                logger.warning(f"config with problem: {config}")
                return False

        # Check switch attention heads divisibility
        if 'attention' in config.get('switch_head_type', '').lower():
            switch_heads = config.get('switch_attention_heads', 1)
            gnn_hidden_dim = config.get('gnn_hidden_dim', 128)
            if switch_heads > 0 and gnn_hidden_dim % switch_heads != 0:
                logger.warning(f"Invalid attention config: {gnn_hidden_dim} not divisible by {switch_heads} heads")
                logger.warning(f"config with problem: {config}")
                return False

        # Ensure at least one MLP is enabled
        if not (config.get('use_node_mlp') or config.get('use_edge_mlp')):
            logger.warning("Invalid config: Neither node nor edge MLP enabled")
            logger.warning(f"config with problem: {config}")
            return False

        # Gated MP needs a GNN backbone
        if config.get('use_gated_mp') and not config.get('gnn_type'):
            logger.warning("Invalid config: Gated MP requires GNN")
            logger.warning(f"config with problem: {config}")
            return False

        # PhyR dependency
        if config.get('use_phyr') and 'phyr_k_ratio' not in config:
            logger.warning("Invalid config: PhyR enabled but no k_ratio specified")
            logger.warning(f"config with problem: {config}")
            return False

        return True
    


    def objective(self, trial: optuna.Trial) -> float:
        """The all-in-one objective function for a single HPO trial."""
        global WORKER_DATALOADERS
        
        # Force CPU and limit threads in each worker process
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cpu':
            torch.set_num_threads(1)
        
        if self.seed ==0:
            seed = trial.number
        else:
            seed = self.seed
        set_global_determinism(seed)

        # Load data only if not already loaded in this worker process
        if not WORKER_DATALOADERS:
            logger.info(f"Worker (PID: {os.getpid()}) loading data for the first time.")
            dataloaders = create_data_loaders(
                dataset_names=self.config.get('dataset_names'),
                folder_names=self.config.get('folder_names'),
                batch_size=self.fixed_params.get('batch_size', 128),
                seed=self.seed,
                num_workers=0, 
            )
            WORKER_DATALOADERS['train'] = dataloaders.get("train")
            WORKER_DATALOADERS['val'] = dataloaders.get("validation")

        train_loader = WORKER_DATALOADERS['train']
        val_loader = WORKER_DATALOADERS['val']

        data_sample = train_loader.dataset[0] if train_loader else None

        config = self.suggest_params(trial)
        initial_model_state_dict = None

        if not self._validate_config(config):
            logger.warning(f"Trial {trial.number} skipped due to invalid configuration: {config}")
            raise optuna.exceptions.TrialPruned("Invalid configuration")

        if self.stage1_checkpoint:
            logger.info(f"Stage 2: Loading checkpoint {self.stage1_checkpoint}")
            checkpoint = torch.load(self.stage1_checkpoint, map_location=device)

            # 1) pull out *just* the architecture kwargs
            arch = checkpoint.get('stage1_architecture',
                {k: checkpoint['config'][k]
                    for k in MODEL_KWARGS_KEYS
                    if k in checkpoint['config']})

            model_kwargs = {
                'node_input_dim': data_sample.x.shape[1],
                'edge_input_dim': data_sample.edge_attr.shape[1],
                **arch
            }
            model = AdvancedMLP(**model_kwargs).to(device)

            # 2) load the *initial* random weights, if available, else fall back
            init_state = checkpoint.get('initial_model_state_dict',
                                        checkpoint['model_state_dict'])
            model.load_state_dict(init_state, strict=True)
            logger.info("✅ Loaded stage-1 initial weights for stage-2 HPO")
        else:
            # --- STAGE 1: Initialize model from scratch ---
            model_kwargs = {
                'node_input_dim': data_sample.x.shape[1],
                'edge_input_dim': data_sample.edge_attr.shape[1],
                **config  
            }
            model = AdvancedMLP(**model_kwargs).to(device)
            # Save the random initial state for reproducibility
            initial_model_state_dict = model.state_dict()

        model_params = sum(p.numel() for p in model.parameters())

        if model_params > 2e6:
            logger.warning(f"Model has {model_params} parameters, which is quite large")
            raise optuna.exceptions.TrialPruned("Model too large")

        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        crit_name = config.get('criterion_name', 'FocalLoss')
        if crit_name == 'FocalLoss':
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif crit_name == 'WeightedBCELoss':
            criterion = WeightedBCELoss()
        elif crit_name == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss: {crit_name}")
        
        lambda_dict = {k: config[k] for k in config if k.startswith('lambda_')}

        # 4. Training loop with pruning and timeout
        start_time = time.time()
        best_mcc_in_trial = -1.0
        best_model_state_dict = None
        best_epoch = -1
        
        max_epochs = min(config.get('epochs', 100), 60)
        for epoch in range(max_epochs):
            if time.time() - start_time > self.trial_timeout:
                raise optuna.exceptions.TrialPruned("Trial timed out")

            train_loss, train_dict = train(model, train_loader, optimizer, criterion, device, **lambda_dict)
            val_loss, val_dict = test(model, val_loader, criterion, device, **lambda_dict)

            # logger.info(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            # if epoch % 5 == 0:
            #     logger.info(f"Epoch {epoch+1}/{max_epochs}, Train dict:{train_dict}, \n \n Val dict: {val_dict}")

            current_mcc = val_dict.get('test_mcc', -1.0)

            # If the current model is the best one so far in this trial, save its state
            if current_mcc > best_mcc_in_trial:
                best_mcc_in_trial = current_mcc
                best_epoch = epoch + 1
                best_model_state_dict = model.state_dict()
            
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if best_model_state_dict is not None:
            mcc_threshold = 0.4
            should_save = False
            top_mccs = trial.study.user_attrs.get("top_mccs", [])

            # Condition 1: Is MCC above the absolute threshold?
            if best_mcc_in_trial > mcc_threshold:
                should_save = True
            elif len(top_mccs) < 5 or best_mcc_in_trial > min(top_mccs):
                if any(mcc > mcc_threshold for mcc in top_mccs):
                    should_save = True

            if should_save:
                model_path = self.results_dir / f"model_trial_{trial.number}.pt"
                torch.save({
                    'model_state_dict': best_model_state_dict,
                    'initial_model_state_dict': initial_model_state_dict,
                    'config': config,
                    'seed': seed,
                    'best_mcc_in_trial': best_mcc_in_trial,
                    'best_epoch': best_epoch,
                }, model_path)
                logger.info(f"Saved model from trial {trial.number} to {model_path} (MCC: {best_mcc_in_trial:.4f})")

                # Update the list of top MCCs
                top_mccs.append(best_mcc_in_trial)
                top_mccs = sorted(top_mccs, reverse=True)[:5]
                trial.study.set_user_attr("top_mccs", top_mccs)
            else:
                logger.info(f"Discarded model from trial {trial.number} (MCC: {best_mcc_in_trial:.4f}). Does not meet save criteria.")

        # 5. Log all metrics and return the objective value
        final_metrics = {
            'best_mcc': best_mcc_in_trial,
            'final_val_loss': val_loss,
            'final_epoch': epoch,
            'model_parameters': model_params,
        }
        trial.set_user_attr("metrics", final_metrics)
        trial.set_user_attr("hyperparameters", config) 

        return best_mcc_in_trial

    def run_async_optimization(self, n_trials: int = 100, use_wandb: bool = False):
        """Run asynchronous parallel optimization with fixed wandb support"""
        batch_size = self.n_parallel  

        if use_wandb:
            self.wandb_run = wandb.init(
                project=self.config.get('wandb_project', 'HPO'),
                name=self.experiment_id,
                job_type="parallel_hpo_experiment",
                config={**self.config, 'n_parallel': self.n_parallel, 'processing_batch_size': batch_size}
            )

        logger.info(f"Starting Parallel HPO: {self.experiment_id}")
        logger.info(f"Parallel workers: {self.n_parallel}, Processing batch size: {batch_size}")
        
        # Create study with async-compatible sampler - suppress warnings
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=self.n_startup_trials,
            multivariate=True,  
            warn_independent_sampling=False
        )
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.pruner_startup,
                n_warmup_steps=self.pruner_warmup,
            )
        )
        study.set_user_attr("top_mccs", [])

        logger.info("Dataloaders will be created once per worker process.")

        # Run optimization. The objective function now handles its own data loading.
        study.optimize(
            self.objective, 
            n_trials=n_trials,
            n_jobs=self.n_parallel,
            catch=(Exception,),
            callbacks=[self._log_trial_to_csv_callback]
        )

        # Log final results to wandb if enabled
        if use_wandb and self.wandb_run:
            best_value = study.best_value if study.trials else -1
            self.wandb_run.log({
                "experiment/final_best_mcc": best_value,
                "experiment/total_trials": len(study.trials),
                "experiment/completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            })
            self.wandb_run.finish()
        
        # Save results
        self._save_results(study)
        
        return study

    def _save_results(self, study: optuna.Study):
        """Save final artifacts using the CSV file generated by the callback."""
        logger.info("\n--- HPO Experiment Complete ---")

        # 1. Read the final CSV file created by the callback
        if not (self.csv_file.exists() and self.csv_file.stat().st_size > 0):
            logger.error("CSV file is empty or missing. Cannot generate final report.")
            return

        try:
            df = pd.read_csv(self.csv_file, quoting=1)
            logger.info(f"Results successfully loaded from: {self.csv_file}")
        except Exception as e:
            logger.error(f"Failed to read final results from CSV: {e}")
            return
        
        # 2. Find the best trial from the CSV data
        completed_df = df[df['state'] == 'COMPLETE'].copy()
        if completed_df.empty:
            logger.warning("No trials completed successfully.")
            best_mcc, best_trial_num = -1, -1
        else:
            best_row = completed_df.loc[completed_df['value'].idxmax()]
            best_mcc = best_row['value']
            best_trial_num = int(best_row['trial_number'])

        # 3. Save the best configuration from Optuna's study object
        if study.best_trial:
            # We still get the best config from the study object for accuracy
            best_config = study.best_trial.user_attrs.get("hyperparameters", study.best_params)
            best_config['_hpo_metadata'] = {
                'best_mcc_from_csv': best_mcc,
                'best_trial_number': best_trial_num,
                'total_trials_in_csv': len(df),
            }
            config_file = self.results_dir / "best_config.yaml"
            with open(config_file, 'w', encoding="utf-8") as f:
                yaml.dump(best_config, f)
            logger.info(f"Best config saved to: {config_file}")

        # 4. Create and save plots
        fig = self.create_parallel_plot_from_dataframe(completed_df)
        if fig:
            plot_file = self.results_dir / "parallel_coordinates.html"
            fig.write_html(plot_file)
            logger.info(f"Interactive plot saved: {plot_file}")

        # 5. Log final summary
        logger.info(f"Best MCC Score: {best_mcc:.4f} (Trial #{best_trial_num})")
        if not completed_df.empty:
            display_cols = ['trial_number', 'value', 'final_val_loss', 'final_epoch']
            logger.info("Top 5 Completed Trials from CSV:")
            logger.info("\n" + completed_df.nlargest(5, 'value')[display_cols].to_string(index=False))

    def create_parallel_plot(self, study: optuna.Study) -> go.Figure:
        """Create parallel coordinates plot with robust CSV handling"""
        try:
            if self.csv_file.exists() and self.csv_file.stat().st_size > 0:
                try:
                    df = pd.read_csv(self.csv_file, quoting=1, on_bad_lines='skip')
                    df = df[df['status'] == 'completed'].sort_values('best_mcc', ascending=False)  
                    
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
    
    def create_parallel_plot_from_dataframe(self, df: pd.DataFrame, eval_metric: str = 'best_mcc') -> go.Figure:
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
                       'best_balanced_accuracy', 'final_balanced_accuracy', "datetime_start",
                       "datetime_complete", "elapsed_time", "wandb_run_id", "wandb_run", "value"}
        
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


def main():
    # Set proper multiprocessing start method for cluster
    mp.set_start_method('fork', force=True)
    
    parser = argparse.ArgumentParser(description="Simplified HPO")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--study_name", type=str, default="hpo", help="Study name")
    parser.add_argument("--n_parallel", type=int, default=2, help="Number of parallel trials")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--wandb", action="store_true", help="Use W&B")
    parser.add_argument("--trial_timeout", type=int, default=3600, help="Timeout per trial in seconds (default: 1 hour)")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to stage 1 checkpoint for resuming HPO")
    args = parser.parse_args()
    
    config_path = Path("model_search/config_files") / args.config
    hpo = HPO(config_path, args.study_name, n_parallel=args.n_parallel, 
              trial_timeout=args.trial_timeout, stage1_checkpoint=args.stage1_checkpoint)
    hpo.run_async_optimization(args.trials, args.wandb)


if __name__ == "__main__":
    main()