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
    def __init__(self,  config_path: str, study_name: str):
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
    
    def _init_csv_tracking(self):
        self.csv_file.touch()
        
    def _log_trial_to_csv(self, trial_num: int, config: dict, metrics: dict):
        """Log trial results to CSV with proper handling of list parameters"""
        row_data = {
            'trial_number': trial_num,
            'timestamp': datetime.now().isoformat(),
            **metrics  
        }
        
        # Add config parameters, handling lists properly
        for key, value in config.items():
            if isinstance(value, list):
                row_data[key] = str(value).replace(',', ';')
            elif isinstance(value, (dict, tuple)):
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
        """Suggest parameters with feasibility checks"""
        config = self.config.copy()
        config.update(self.fixed_params)
        
        # Suggest all parameters first
        for param, spec in self.search_space.items():
            if spec['search_type'] == 'float':
                config[param] = trial.suggest_float(param, spec['min'], spec['max'], 
                                                  log=spec.get('log', False))
            elif spec['search_type'] == 'int':
                config[param] = trial.suggest_int(param, spec['min'], spec['max'], 
                                                step=spec.get('step', 1))
            elif spec['search_type'] == 'categorical':
                config[param] = trial.suggest_categorical(param, spec['choices'])
            elif spec['search_type'] == 'dynamic_list':
                config[param] = self._suggest_dynamic_list(trial, param, spec)
        
        # Apply feasibility constraints
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
        
        # Constraint 9: 
        if not config.get('use_node_mlp') and not config.get('use_edge_mlp'):
            config['use_node_mlp'] = True
            if 'node_hidden_dims' not in config:
                config['node_hidden_dims'] = [128]
        
        
        if config.get('use_gated_mp') and not config.get('gnn_type'):
            config['use_gated_mp'] = False
            logger.debug("Disabled gated MP because no GNN type specified")
        
        
        if not config.get('use_phyr', False):
            config.pop('phyr_k_ratio', None)
        
        output_type = config.get('output_type', 'binary')
        if output_type == 'multiclass' and 'criterion_name' not in self.fixed_params:
            config['criterion_name'] = 'CrossEntropyLoss'
        elif output_type == 'binary' and 'criterion_name' not in self.fixed_params:
            config['criterion_name'] = 'BCEWithLogitsLoss'
        elif output_type == 'regression' and 'criterion_name' not in self.fixed_params:
            config['criterion_name'] = 'MSELoss'
        
        
        if output_type == 'multiclass':
            config['num_classes'] = 2  
        elif output_type == 'binary':
            config['num_classes'] = 2
        else:  
            config.pop('num_classes', None)
        
    
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
            final_train_loss = float('inf')
            final_val_loss = float('inf')
            final_precision = 0.0
            final_recall = 0.0
            final_accuracy = 0.0
            
            patience = 0
            max_epochs = min(config.get('epochs', 100), 50)
            max_patience = config.get('patience', 8)
            
            for epoch in range(max_epochs):
                train_loss, train_dict = train(model, train_loader, optimizer, criterion, self.device)
                val_loss, val_dict = test(model, val_loader, criterion, self.device)
                
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
    
    def _setup_training(self, config: dict):
        """Setup model, data, and criterion"""
        try:
            # Load model
            model_module = importlib.import_module(f"models.{config['model_module']}.{config['model_module']}")
            model_class = getattr(model_module, config['model_module'])
            
            # Create data loaders
            dataloaders = create_data_loaders(
                dataset_names=config['dataset_names'],
                folder_names=config['folder_names'],
                dataset_type=config.get('dataset_type', 'default'),
                batch_size=config['batch_size'],
                max_nodes=config.get('max_nodes', 1000),
                max_edges=config.get('max_edges', 5000),
                train_ratio=config.get('train_ratio', 0.85),
                seed=config.get('seed', 42),
                num_workers=config.get('num_workers', 0),
                batching_type=config.get('batching_type', 'dynamic'),
            )
            
            train_loader = dataloaders.get("train")
            val_loader = dataloaders.get("validation")
            
            if not train_loader or not val_loader:
                return None, None, None, None
            
            # Get dimensions and create model
            sample = train_loader.dataset[0]
            model_kwargs = {
                'node_input_dim': sample.x.shape[1],
                'edge_input_dim': sample.edge_attr.shape[1],
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
                       'final_recall', 'final_accuracy', 'final_epoch', 'converged', 
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
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=10),
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15)
        )
        
        # Add callback for experiment-level logging
        def experiment_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE and self.wandb_run:
                completed_trials = len([t for t in study.trials 
                                      if t.state == optuna.trial.TrialState.COMPLETE])
                best_f1 = 1 - study.best_value if study.best_value < 1 else 0
                
                # Log experiment-level metrics
                wandb.log({
                    "experiment/completed_trials": completed_trials,
                    "experiment/best_f1_so_far": best_f1,
                    "experiment/trial_number": trial.number,
                    "experiment/current_trial_f1": trial.user_attrs.get('best_f1_score', 0),
                })
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, callbacks=[experiment_callback])
        
        # Final experiment summary
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if self.wandb_run and completed:
            # Log final experiment summary
            all_f1s = [t.user_attrs.get('best_f1_score', 0) for t in completed]
            wandb.log({
                "experiment/final_best_f1": max(all_f1s),
                "experiment/mean_f1": np.mean(all_f1s),
                "experiment/std_f1": np.std(all_f1s),
                "experiment/total_completed": len(completed),
                "experiment/success_rate": len(completed) / len(study.trials)
            })
        
        # Save results
        self._save_results(study)
        
        if self.wandb_run:
            wandb.finish()
        
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
                    # Select columns for top 5 display, including per-class accuracies and loss progression
                    display_cols = ['trial_number', 'best_f1_score', 'starting_loss', 'best_val_loss', 'final_epoch']
                    
                    # Add per-class accuracy columns if they exist
                    class_cols = [col for col in completed_df.columns if 'class_' in col and 'accuracy' in col]
                    if class_cols:
                        display_cols.extend(class_cols)
                    
                    # Add overall accuracy if available
                    if 'final_accuracy' in completed_df.columns:
                        display_cols.append('final_accuracy')
                    
                    # Add loss improvement column if starting_loss exists
                    if 'starting_loss' in completed_df.columns:
                        completed_df['loss_improvement'] = completed_df['starting_loss'] - completed_df['best_val_loss']
                        display_cols.append('loss_improvement')
                    
                    top_5 = completed_df.nlargest(5, 'best_f1_score')[display_cols]
                    logger.info(f"\nTop 5 Trials from CSV:")
                    logger.info(top_5.to_string(index=False))
                    
                    # Additional per-class summary for best trial
                    best_trial = completed_df.loc[completed_df['best_f1_score'].idxmax()]
                    if class_cols:
                        logger.info(f"\nBest Trial ({int(best_trial['trial_number'])}) Per-Class Performance:")
                        for col in class_cols:
                            class_name = col.replace('class_', '').replace('_accuracy', '')
                            accuracy = best_trial[col]
                            if pd.notna(accuracy):
                                logger.info(f"  Class {class_name}: {accuracy:.1%}")
                    
                    # Loss progression summary for best trial
                    if 'starting_loss' in best_trial and pd.notna(best_trial['starting_loss']):
                        start_loss = best_trial['starting_loss']
                        best_loss = best_trial['best_val_loss']
                        improvement = start_loss - best_loss
                        improvement_pct = (improvement / start_loss) * 100
                        logger.info(f"\nBest Trial Loss Progression:")
                        logger.info(f"  Starting Loss: {start_loss:.4f}")
                        logger.info(f"  Best Val Loss: {best_loss:.4f}")
                        logger.info(f"  Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")
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
                starting_loss = trial.user_attrs.get('starting_loss', 'N/A')
                epoch = trial.user_attrs.get('final_epoch', 'N/A')
                accuracy = trial.user_attrs.get('final_accuracy', 'N/A')
                
                # Calculate loss improvement if both values available
                loss_improvement = ""
                if starting_loss != 'N/A' and val_loss != 'N/A':
                    improvement = starting_loss - val_loss
                    loss_improvement = f", Δ={improvement:.4f}"
                
                # Per-class accuracies from Optuna user_attrs
                class_info = ""
                for key, value in trial.user_attrs.items():
                    if 'class_' in key and 'accuracy' in key:
                        class_name = key.replace('class_', '').replace('_accuracy', '')
                        class_info += f", Class_{class_name}={value:.1%}"
                
                logger.info(f"  Trial {trial.number}: F1={f1:.4f}, Start={starting_loss}, Val={val_loss}{loss_improvement}, Epoch={epoch}{class_info}")
                
            # Show best trial per-class summary
            if top_trials:
                best_trial = top_trials[0]
                class_accuracies = {k: v for k, v in best_trial.user_attrs.items() 
                                  if 'class_' in k and 'accuracy' in k}
                if class_accuracies:
                    logger.info(f"\nBest Trial ({best_trial.number}) Per-Class Performance:")
                    for key, accuracy in class_accuracies.items():
                        class_name = key.replace('class_', '').replace('_accuracy', '')
                        logger.info(f"  Class {class_name}: {accuracy:.1%}")
                
                # Loss progression for best trial
                starting_loss = best_trial.user_attrs.get('starting_loss')
                best_val_loss = best_trial.user_attrs.get('best_val_loss')
                if starting_loss is not None and best_val_loss is not None:
                    improvement = starting_loss - best_val_loss
                    improvement_pct = (improvement / starting_loss) * 100
                    logger.info(f"\nBest Trial Loss Progression:")
                    logger.info(f"  Starting Loss: {starting_loss:.4f}")
                    logger.info(f"  Best Val Loss: {best_val_loss:.4f}")
                    logger.info(f"  Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")


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