import argparse
from typing import final, Dict, Any
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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time
from functools import partial
import json

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

class HPO:
    """Parallelized HPO with multi-process execution and async TPE updates"""
    
    def __init__(self, config_path: str, study_name: str, startup_trials: int = 25,
                 pruner_startup: int = 15, pruner_warmup: int = 15, n_parallel: int = None):
        self.n_startup_trials = startup_trials
        self.pruner_startup = pruner_startup
        self.pruner_warmup = pruner_warmup
        self.seed = 2
        self.config_path = config_path
        self.study_name = study_name
        
        # Force CPU usage
        self.device = torch.device("cpu")
        torch.set_num_threads(1)  # Prevent thread oversubscription
        
        # Set parallel workers based on CPU cores
        if n_parallel is None:
            n_cpus = mp.cpu_count()
            # Use half the available cores to avoid oversubscription
            self.n_parallel = max(2, min(n_cpus // 2, 8))
        else:
            self.n_parallel = n_parallel
            
        logger.info(f"Using {self.n_parallel} parallel workers on CPU")
        
        with open(config_path, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.search_space = self.config.pop('search_space', {})
        self.fixed_params = self.config.pop('fixed_params', {})
        
        # Results directory
        model_name = self.config.get('model_module', 'Unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path(f"model_search/models/{model_name}/hpo_{study_name}_{timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV tracking with thread safety
        self.csv_file = self.results_dir / "hpo_results.csv"
        self.csv_lock = threading.Lock()
        self._init_csv_tracking()
        
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
        
    def _log_trial_to_csv(self, trial_num: int, config: dict, metrics: dict):
        """Thread-safe CSV logging - now logs ALL parameters including defaults"""
        with self.csv_lock:
            # Combine metrics and full config
            row_data = {
                'trial_number': trial_num,
                'timestamp': datetime.now().isoformat(),
                **metrics
            }
            
            # Add ALL config parameters, including those with default values
            for key, value in config.items():
                if isinstance(value, list):
                    if not value:
                        row_data[key] = "[]"
                    else:
                        row_data[key] = str(value).replace(',', ';')
                elif isinstance(value, (dict, tuple)):
                    row_data[key] = str(value).replace(',', ';')
                elif value is None:
                    row_data[key] = "None"
                elif isinstance(value, bool):
                    row_data[key] = str(value)
                else:
                    row_data[key] = value
            
            df = pd.DataFrame([row_data])
            
            if not self.csv_file.exists() or self.csv_file.stat().st_size == 0:
                df.to_csv(self.csv_file, index=False, quoting=1) 
            else:
                df.to_csv(self.csv_file, mode='a', header=False, index=False, quoting=1)
    
    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Suggest parameters with FIXED feasibility checks using YAML config"""
        config = self.config.copy()
        config.update(self.fixed_params)
        
        # Remove any existing GAT/GIN params to ensure clean slate
        for p in [
            'gat_heads','gat_dropout','gat_v2','gat_edge_dim',
            'gin_layers','gin_hidden_dim','gin_eps'
        ]:
            config.pop(p, None)
            
        logger.info("\n inside suggest_params \n")
        logger.info("gnn_type: %s", config.get('gnn_type', 'NONE'))

        # Phase 1: GAT params only if using GAT, else set defaults
        if config.get('gnn_type') == 'GAT':
            logger.info("Using GAT parameters")
            if 'gat_heads' not in config and 'gat_heads' in self.search_space:
                spec = self.search_space['gat_heads']
                if spec['search_type'] == 'categorical':
                    config['gat_heads'] = trial.suggest_categorical('gat_heads', spec['choices'])
                else:
                    config['gat_heads'] = trial.suggest_int('gat_heads', spec['min'], spec['max'])
            if 'gat_dropout' not in config and 'gat_dropout' in self.search_space:
                spec = self.search_space['gat_dropout']
                config['gat_dropout'] = trial.suggest_float('gat_dropout', spec['min'], spec['max'], log=spec.get('log', False))
            if 'gat_v2' not in config and 'gat_v2' in self.search_space:
                spec = self.search_space['gat_v2']
                config['gat_v2'] = trial.suggest_categorical('gat_v2', spec['choices'])
            if 'gat_edge_dim' not in config and 'gat_edge_dim' in self.search_space:
                spec = self.search_space['gat_edge_dim']
                config['gat_edge_dim'] = trial.suggest_int('gat_edge_dim', spec['min'], spec['max'])
        else:
            config['gat_heads']    = 0
            config['gat_dropout']  = 0.0
            config['gat_v2']       = False
            config['gat_edge_dim'] = 0

        # Phase 2: GIN params only if using GIN, else set defaults
        if config.get('gnn_type') == 'GIN':
            if 'gin_layers' not in config and 'gin_layers' in self.search_space:
                spec = self.search_space['gin_layers']
                config['gin_layers'] = trial.suggest_int('gin_layers', spec['min'], spec['max'])
            if 'gin_hidden_dim' not in config and 'gin_hidden_dim' in self.search_space:
                spec = self.search_space['gin_hidden_dim']
                config['gin_hidden_dim'] = trial.suggest_int('gin_hidden_dim', spec['min'], spec['max'])
            if 'gin_eps' not in config and 'gin_eps' in self.search_space:
                spec = self.search_space['gin_eps']
                config['gin_eps'] = trial.suggest_float('gin_eps', spec['min'], spec['max'], log=spec.get('log', False))
        else:
            config['gin_layers']     = 0
            config['gin_hidden_dim'] = 0
            config['gin_eps']        = 0.0
        
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
                    valid_choices = []
                    for heads in spec['choices']:
                        if gnn_hidden_dim >= heads and gnn_hidden_dim % heads == 0:
                            valid_choices.append(heads)
                    if not valid_choices:
                        valid_choices = [1]  
                config['switch_attention_heads'] = trial.suggest_categorical('switch_attention_heads', valid_choices)
            elif spec['search_type'] == 'int':
                valid_heads = []
                for heads in range(spec['min'], spec['max'] + 1):
                    if gnn_hidden_dim % heads == 0:
                        valid_heads.append(heads)
                if not valid_heads:
                    valid_heads = [1]  
                config['switch_attention_heads'] = trial.suggest_categorical('switch_attention_heads_valid', valid_heads)
        elif 'attention' not in config.get('switch_head_type', '').lower():
            # If not using attention, set to 0
            config['switch_attention_heads'] = 0
        
        # Phase 5: Process all other parameters with proper separation
        for param, spec in self.search_space.items():
            if param in config:
                continue  # Already handled above
            
            # Handle boolean flags BEFORE their associated parameters
            if param in ['use_node_mlp', 'use_edge_mlp'] and spec['search_type'] == 'categorical':
                config[param] = trial.suggest_categorical(param, spec['choices'])
                continue
                
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
                # Only suggest dynamic lists for the actual list parameters
                if param in ['node_hidden_dims', 'edge_hidden_dims']:
                    # Check if the corresponding MLP is enabled
                    if param == 'node_hidden_dims' and config.get('use_node_mlp', True):
                        config[param] = self._suggest_dynamic_list(trial, param, spec)
                    elif param == 'edge_hidden_dims' and config.get('use_edge_mlp', True):
                        config[param] = self._suggest_dynamic_list(trial, param, spec)
                    # If MLP is disabled, don't set the hidden_dims at all
                else:
                    config[param] = self._suggest_dynamic_list(trial, param, spec)
        
        # Phase 6: Final constraint validation and fixes
        config = self._apply_constraints(trial, config)
        
        # Phase 7: Ensure boolean parameters remain boolean
        for bool_param in ['use_node_mlp', 'use_edge_mlp']:
            if bool_param in config and not isinstance(config[bool_param], bool):
                # If somehow a non-boolean value got assigned, fix it
                logger.warning(f"Non-boolean value {config[bool_param]} found for {bool_param}, converting to boolean")
                config[bool_param] = bool(config[bool_param])

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
    
    def _apply_constraints(self, trial: optuna.Trial, config: dict) -> dict:
        """Apply comprehensive feasibility constraints with standardized null values"""
        
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
        # 1) build config
        config = self.suggest_params(trial)
        # 2) run exactly what _trial_worker does, but in‐process
        result = _trial_worker(
            str(self.config_path),
            self.study_name,
            trial._trial_id,
            config,
            self.seed,
            self.config,
            self.fixed_params
        )
        # 3) push metrics into trial attrs
        for k, v in result['metrics'].items():
            trial.set_user_attr(k, v)
        # 4) return the objective
        return result['objective_value']
    def run_parallel_batch(self, study: optuna.Study, batch_size: int = None) -> list:
        """Run a batch of trials in parallel - FIXED version"""
        if batch_size is None:
            batch_size = self.n_parallel
        
        # Generate batch of trials with complete configs
        trials_with_configs = []
        for _ in range(batch_size):
            trial = study.ask()
            # Generate complete config including defaults HERE in main process
            complete_config = self.suggest_params(trial)
            trials_with_configs.append((trial, complete_config))
        
        # Run trials in parallel
        with ProcessPoolExecutor(max_workers=self.n_parallel) as executor:
            # Submit all trials with their complete configs
            future_to_trial = {}
            for trial, config in trials_with_configs:
                future = executor.submit(
                    _trial_worker,
                    str(self.config_path),      
                    self.study_name,           
                    trial._trial_id,           # Pass trial ID instead of trial object
                    config,                    # Pass the complete config
                    self.seed,
                    self.config,
                    self.fixed_params
                )
                future_to_trial[future] = (trial, config)
            
            # Collect results as they complete
            results = []
            for future in as_completed(future_to_trial):
                trial, config = future_to_trial[future]
                try:
                    result = future.result()
                    for key, value in result['metrics'].items():
                        trial.set_user_attr(key, value)
                    study.tell(trial, result['objective_value'], state=optuna.trial.TrialState.COMPLETE)
                    results.append((trial, result))


                    self._log_trial_to_csv(trial._trial_id, config, result['metrics'])
                    
                except Exception as e:
                    logger.error(f"Trial {trial._trial_id} failed: {e}")
                    # Always mark the trial as failed
                    try:
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    except Exception:
                        pass

                    
                    # Log failed trial to CSV
                    failed_metrics = {
                        'status': 'failed',
                        'error': str(e),
                        'best_mcc': -1.0,
                        'best_val_loss': float('inf'),
                        'final_epoch': 0,
                        'model_parameters': 0
                    }
                    self._log_trial_to_csv(trial._trial_id, config, failed_metrics)
                    results.append((trial, {'objective_value': float('-inf'), 'error': str(e)}))
        
        return results

    def run_async_optimization(self, n_trials: int = 100, use_wandb: bool = False):
        """Run asynchronous parallel optimization"""
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
        
        # Create study with async-compatible sampler
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=self.n_startup_trials,
            multivariate=True,  # Better for parallel execution
        )
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.pruner_startup,
                n_warmup_steps=self.pruner_warmup,
            )
        )

        # completed_trials = 0
        # start_time = time.time()
        
        # while completed_trials < n_trials:
        #     remaining_trials = n_trials - completed_trials
        #     current_batch_size = min(batch_size, remaining_trials)
            
        #     logger.info(f"Running batch of {current_batch_size} trials ({completed_trials}/{n_trials} completed)")
            
        #     batch_results = self.run_parallel_batch(study, current_batch_size)
        #     completed_trials += len(batch_results)
            
        #     # Log progress
        #     if self.wandb_run:
        #         best_value = study.best_value if study.trials else -1
        #         elapsed_time = time.time() - start_time
        #         wandb.log({
        #             "experiment/completed_trials": completed_trials,
        #             "experiment/best_mcc_so_far": best_value,
        #             "experiment/elapsed_time": elapsed_time,
        #             "experiment/trials_per_minute": completed_trials / (elapsed_time / 60)
        #         })
            
        #     logger.info(f"Batch completed. Best MCC so far: {study.best_value:.4f}")

        # # Save results
        # self._save_results(study)
        
        # if self.wandb_run:
        #     wandb.finish()
        
        # return study
        # fire n_trials asynchronously on n_parallel workers:
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=self.n_parallel,
            catch=(Exception,),
        )
        # once done, save & exit
        self._save_results(study)
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
        
        # Save best config
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

        # Create parallel plot
        fig = self.create_parallel_plot(study)
        if fig:
            plot_file = self.results_dir / "parallel_coordinates.html"
            fig.write_html(plot_file)
            logger.info(f"Interactive plot saved: {plot_file}")

        # Save study
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

        # Show top trials
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


def _trial_worker(config_path: str, study_name: str, trial_id: int, 
                  complete_config: Dict[str, Any], seed: int, 
                  base_config: dict, fixed_params: dict) -> dict:
    """Worker function that runs in separate process - receives complete config"""
    try:
        # Force CPU and limit threads in worker
        device = torch.device("cpu")
        torch.set_num_threads(1)
        
        # Set environment variables for this worker
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # The config is already complete with all defaults applied
        config = complete_config
        
        # Load model module
        model_module = importlib.import_module(f"models.{config['model_module']}.{config['model_module']}")
        model_class = getattr(model_module, config['model_module'])
        
        # Create data loaders for this worker
        batch_size = config.get('batch_size', 128)
        dataloaders = create_data_loaders(
            dataset_names=config.get('dataset_names', base_config['dataset_names']),
            folder_names=config.get('folder_names', base_config['folder_names']),
            dataset_type=config.get('dataset_type', 'default'),
            batch_size=batch_size,
            max_nodes=config.get('max_nodes', 1000),
            max_edges=config.get('max_edges', 5000),
            train_ratio=config.get('train_ratio', 0.85),
            seed=seed,
            num_workers=0,  # Important: Set to 0 for multiprocessing
            batching_type=config.get('batching_type', 'dynamic'),
        )
        
        train_loader = dataloaders.get("train")
        val_loader = dataloaders.get("validation")
        
        if not train_loader or not val_loader:
            return {'objective_value': float('-inf'), 'metrics': {'status': 'failed_data_loading'}}

        data_sample = train_loader.dataset[0]
        
        # Build model kwargs
        model_kwargs = {
            'node_input_dim': data_sample.x.shape[1],
            'edge_input_dim': data_sample.edge_attr.shape[1],
        }

        # Add all model parameters from config
        for key in ['activation', 'dropout_rate', 'gnn_type', 'gnn_layers', 'gnn_hidden_dim', 
            'gat_heads', 'gat_dropout', 'gin_eps', 'use_node_mlp', 'use_edge_mlp',
            'node_hidden_dims', 'edge_hidden_dims', 'use_batch_norm', 'use_residual',
            'use_skip_connections', 'switch_head_type', 'switch_head_layers', 
            'switch_attention_heads', 'output_type', 'num_classes', 'use_gated_mp',
            'use_phyr', 'phyr_k_ratio', 'pooling', 'normalization_type', 
            'loss_scaling_strategy', 'enforce_radiality']:
            if key in config:
                value = config[key]
                
                # Skip empty lists or zero values for optional parameters
                if key == 'node_hidden_dims' and (not value or value == []):
                    if config.get('use_node_mlp', True):
                        model_kwargs[key] = [128]  # Default fallback
                    else:
                        continue  
                elif key == 'edge_hidden_dims' and (not value or value == []):
                    if config.get('use_edge_mlp', True):
                        model_kwargs[key] = [128]  # Default fallback
                    else:
                        continue  
                elif key in ['gat_heads', 'gat_dropout', 'gin_eps', 'gin_mlp_layers', 
                            'switch_attention_heads', 'phyr_k_ratio'] and value == 0:
                    continue
                elif key == 'gin_train_eps' and not config.get('gnn_type') == 'GIN':
                    continue
                elif key == 'gnn_hidden_dim' and value == 0:
                    continue  
                else:
                    model_kwargs[key] = value
        
        model = model_class(**model_kwargs).to(device)
        model_params = sum(p.numel() for p in model.parameters())

        # Setup criterion
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
        
        # Training loop
        starting_val_loss = float('inf')
        starting_train_loss = float('inf')
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        best_minority_f1 = 0.0
        best_mcc = -1.0 
        best_balanced_accuracy = 0.0
        patience = 0
        max_epochs = min(config.get('epochs', 100), 80)
        max_patience = config.get('patience', 25)
        
        for epoch in range(max_epochs):
            train_loss, train_dict = train(model, train_loader, optimizer, criterion, device, **lambda_dict)
            val_loss, val_dict = test(model, val_loader, criterion, device, **lambda_dict)
            print(f"Epoch: {epoch}  ->      train loss: {train_loss},       val loss: {val_loss}")
            if epoch == 0: 
                starting_val_loss = val_loss
                starting_train_loss = train_loss
                
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
            
            # Early stopping
            if patience >= max_patience:
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
        
        return {'objective_value': best_mcc, 'metrics': metrics}
        
    except Exception as e:
        logger.error(f"Trial {trial_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return {'objective_value': float('-inf'), 'metrics': {'status': 'failed', 'error': str(e)}}


def main():
    # Set proper multiprocessing start method for cluster
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Simplified HPO")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--study_name", type=str, default="hpo", help="Study name")
    parser.add_argument("--n_parallel", type=int, default=2, help="Number of parallel trials")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--wandb", action="store_true", help="Use W&B")
    
    args = parser.parse_args()
    
    config_path = Path("model_search/config_files") / args.config
    hpo = HPO(config_path, args.study_name, n_parallel=args.n_parallel)
    hpo.run_async_optimization(args.trials, args.wandb)


if __name__ == "__main__":
    main()