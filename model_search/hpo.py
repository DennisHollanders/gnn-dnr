import argparse
import os
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
import json
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
from datetime import datetime
import pickle
import plotly.graph_objects as go
import pandas as pd

# Setup logging
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter("%(message)s"))   

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("Two-Stage HPO")
logger.setLevel(logging.INFO)
logger.handlers.clear()         
logger.addHandler(console)
logger.propagate = False 

# Add paths
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

model_search_path = ROOT_DIR / "model_search"
if str(model_search_path) not in sys.path:
    sys.path.append(str(model_search_path))

from load_data import create_data_loaders
from train import train, test


class TwoStageHPOTuner:
    """Two-Stage HPO with configurable search space from YAML and W&B tracking"""
    
    def __init__(self, base_config_path: str, study_name: str, stage: int = 1):
        self.base_config_path = base_config_path
        self.study_name = study_name
        self.stage = stage
        self.base_config = self._load_base_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get stage-specific search space
        search_space_key = f'stage_{stage}_search_space'
        if search_space_key not in self.base_config:
            raise ValueError(f"No search space defined for stage {stage}. "
                           f"Please define '{search_space_key}' in config.")
        
        self.search_space = self.base_config[search_space_key]
        
        # Create results directory
        self.model_name = self.base_config.get('model_module', 'Unknown')
        self.results_base_dir = Path(f"model_search/models/{self.model_name}/hpo_{self.study_name}")
        self.results_dir = self.results_base_dir / f"stage_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # W&B run will be initialized later
        self.wandb_run = None
        
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from YAML file"""
        with open(self.base_config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_stage1_results(self) -> Optional[Dict[str, Any]]:
        """Load results from Stage 1 if available"""
        stage1_dirs = list(self.results_base_dir.glob("stage_1_*"))
        if not stage1_dirs:
            return None
        
        # Get most recent Stage 1 results
        latest_stage1 = sorted(stage1_dirs)[-1]
        recommendations_file = latest_stage1 / "stage2_recommendations.json"
        
        if recommendations_file.exists():
            with open(recommendations_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters based on stage configuration"""
        config = self.base_config.copy()
        
        # Remove search space keys from config
        for key in list(config.keys()):
            if key.endswith('_search_space'):
                del config[key]
        
        # Process each parameter in search space
        for param_name, param_spec in self.search_space.items():
            if isinstance(param_spec, dict) and 'search_type' in param_spec:
                search_type = param_spec['search_type']
                
                if search_type == 'float':
                    config[param_name] = trial.suggest_float(
                        param_name,
                        param_spec['min'],
                        param_spec['max'],
                        log=param_spec.get('log', False)
                    )
                elif search_type == 'int':
                    config[param_name] = trial.suggest_int(
                        param_name,
                        param_spec['min'],
                        param_spec['max'],
                        step=param_spec.get('step', 1)
                    )
                elif search_type == 'categorical':
                    config[param_name] = trial.suggest_categorical(
                        param_name,
                        param_spec['choices']
                    )
                elif search_type == 'dynamic':
                    config[param_name] = self._suggest_dynamic_param(
                        trial, param_name, param_spec
                    )
        
        # Handle model-specific parameters
        config = self._suggest_model_specific_params(trial, config)
        
        return config
    
    def _suggest_dynamic_param(self, trial: optuna.Trial, param_name: str, 
                               param_spec: Dict[str, Any]) -> Any:
        """Handle dynamic parameters like variable-length lists"""
        if param_name in ['node_hidden_dims', 'edge_hidden_dims']:
            n_layers = trial.suggest_int(
                f'n_{param_name}_layers',
                param_spec.get('n_layers_min', 1),
                param_spec.get('n_layers_max', 3)
            )
            
            dims = []
            for i in range(n_layers):
                dim = trial.suggest_int(
                    f'{param_name}_{i}',
                    param_spec.get('dim_min', 32),
                    param_spec.get('dim_max', 256),
                    step=param_spec.get('dim_step', 32)
                )
                dims.append(dim)
            
            return dims
        
        return None
    
    def _suggest_model_specific_params(self, trial: optuna.Trial, 
                                      config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model-specific parameter dependencies"""
        model_type = config.get('model_module', 'AdvancedMLP')
        
        if model_type == 'AdvancedMLP':
            # Handle GNN-specific parameters
            if config.get('gnn_type') and config['gnn_type'] != 'None':
                # Ensure GNN parameters are suggested if GNN is used
                if 'gnn_layers' not in config:
                    config['gnn_layers'] = trial.suggest_int('gnn_layers', 1, 3)
                if 'gnn_hidden_dim' not in config:
                    config['gnn_hidden_dim'] = trial.suggest_int(
                        'gnn_hidden_dim', 32, 128, step=32
                    )
                
                if config['gnn_type'] == 'GAT':
                    if 'gat_heads' not in config:
                        config['gat_heads'] = trial.suggest_int('gat_heads', 2, 6)
                    if 'gat_dropout' not in config:
                        config['gat_dropout'] = trial.suggest_float('gat_dropout', 0.0, 0.2)
                elif config['gnn_type'] == 'GIN':
                    if 'gin_eps' not in config:
                        config['gin_eps'] = trial.suggest_float('gin_eps', 0.0, 0.1)
            
            # Ensure at least one MLP is used
            if not config.get('use_node_mlp') and not config.get('use_edge_mlp'):
                config['use_node_mlp'] = True
                if 'node_hidden_dims' not in config:
                    config['node_hidden_dims'] = [128]
        
        return config
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function optimizing F1 score with W&B logging"""
        try:
            config = self.suggest_hyperparameters(trial)
            
            # Log trial start to W&B
            if self.wandb_run:
                wandb.log({
                    f"stage_{self.stage}/trial": trial.number,
                    f"stage_{self.stage}/trial_params": config
                })
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                temp_config_path = f.name
            
            # Load model dynamically
            model_module_path = f"models.{config['model_module']}.{config['model_module']}"
            model_module = importlib.import_module(model_module_path)
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
            validation_loader = dataloaders.get("validation")
            
            if not train_loader or not validation_loader:
                return 1.0
            
            # Get input dimensions
            sample_data = train_loader.dataset[0]
            node_input_dim = sample_data.x.shape[1]
            edge_input_dim = sample_data.edge_attr.shape[1]
            
            # Create model
            model_kwargs = self._prepare_model_kwargs(config, node_input_dim, edge_input_dim)
            model = model_class(**model_kwargs).to(self.device)
            
            # Create criterion
            criterion = self._create_criterion(config, model_module)
            
            # Create optimizer
            optimizer = optim.Adam(
                model.parameters(), 
                lr=config['learning_rate'], 
                weight_decay=config['weight_decay']
            )
            
            # Training loop with W&B logging
            best_f1_score = 0.0
            patience_counter = 0
            max_patience = config.get('patience', 8)
            max_epochs = min(config.get('epochs', 100), 30)  
            
            for epoch in range(max_epochs):
                train_loss, train_dict = train(model, train_loader, optimizer, criterion, self.device)
                val_loss, val_dict = test(model, validation_loader, criterion, self.device)
                
                current_f1 = val_dict.get('test_f1', 0.0)
                if current_f1 == 0.0:
                    precision = val_dict.get('test_precision', 0.0)
                    recall = val_dict.get('test_recall', 0.0)
                    if precision + recall > 0:
                        current_f1 = 2 * precision * recall / (precision + recall)
                
                # Log to W&B
                if self.wandb_run and epoch % 5 == 0:
                    wandb.log({
                        f"stage_{self.stage}/trial_{trial.number}/epoch": epoch,
                        f"stage_{self.stage}/trial_{trial.number}/train_loss": train_loss,
                        f"stage_{self.stage}/trial_{trial.number}/val_loss": val_loss,
                        f"stage_{self.stage}/trial_{trial.number}/f1_score": current_f1,
                        f"stage_{self.stage}/trial_{trial.number}/precision": val_dict.get('test_precision', 0),
                        f"stage_{self.stage}/trial_{trial.number}/recall": val_dict.get('test_recall', 0),
                    })
                
                trial.report(1 - current_f1, epoch)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                if current_f1 > best_f1_score:
                    best_f1_score = current_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        break
                
                if epoch % 5 == 0:
                    logger.debug(f"Trial {trial.number}, Epoch {epoch}: F1={current_f1:.4f}")
            
            # Log final trial results to W&B
            if self.wandb_run:
                wandb.log({
                    f"stage_{self.stage}/trial_{trial.number}/best_f1": best_f1_score,
                    f"stage_{self.stage}/trial_{trial.number}/final_epoch": epoch,
                })
            
            return 1 - best_f1_score  
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            return 1.0
        finally:
            if 'temp_config_path' in locals() and os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
    
    def _prepare_model_kwargs(self, config: Dict[str, Any], node_input_dim: int, 
                             edge_input_dim: int) -> Dict[str, Any]:
        """Prepare model constructor arguments"""
        model_kwargs = {
            'node_input_dim': node_input_dim,
            'edge_input_dim': edge_input_dim,
        }
        
        # Add all relevant parameters from config
        relevant_params = [
            'activation', 'dropout_rate', 'hidden_dims', 'latent_dim',
            'gnn_type', 'gnn_layers', 'gnn_hidden_dim', 'gat_heads', 
            'gat_dropout', 'gin_eps', 'use_node_mlp', 'use_edge_mlp',
            'node_hidden_dims', 'edge_hidden_dims', 'use_batch_norm',
            'use_residual', 'pooling'
        ]
        
        for param in relevant_params:
            if param in config:
                model_kwargs[param] = config[param]
        
        return model_kwargs
    
    def _create_criterion(self, config: Dict[str, Any], model_module) -> nn.Module:
        """Create loss criterion"""
        criterion_name = config.get('criterion_name', 'MSELoss')
        
        try:
            criterion_class = getattr(model_module, criterion_name)
            try:
                criterion = criterion_class(weight_switch=1.0, weight_physics=10.0)
            except:
                criterion = criterion_class()
            return criterion
        except (ImportError, AttributeError):
            return nn.MSELoss()
    
    def create_parallel_coordinates_plot(self, study: optuna.Study, 
                                       title: str = "HPO Results") -> go.Figure:
        """Create interactive parallel coordinates plot"""
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) < 5:
            logger.warning("Too few completed trials for visualization")
            return None
        
        # Create DataFrame
        data = []
        for trial in completed_trials:
            row = trial.params.copy()
            row['F1_Score'] = 1 - trial.value
            row['Trial_Number'] = trial.number
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('F1_Score', ascending=False)
        
        # Select top parameters for visualization
        param_cols = [col for col in df.columns if col not in ['F1_Score', 'Trial_Number']]
        
        # Get parameter importance and select top 8
        try:
            importance = optuna.importance.get_param_importances(study)
            top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]
            param_cols = [param for param, _ in top_params if param in param_cols]
        except:
            param_cols = param_cols[:8]
        
        # Prepare dimensions
        dimensions = [dict(
            label="F1 Score",
            values=df['F1_Score'],
            range=[df['F1_Score'].min(), df['F1_Score'].max()]
        )]
        
        for param in param_cols:
            if df[param].dtype in ['object', 'string', 'bool']:
                # Categorical parameter
                unique_vals = df[param].unique()
                val_to_num = {val: i for i, val in enumerate(unique_vals)}
                numeric_vals = df[param].map(val_to_num)
                
                dimensions.append(dict(
                    label=param,
                    values=numeric_vals,
                    tickvals=list(range(len(unique_vals))),
                    ticktext=[str(val) for val in unique_vals],
                    range=[0, len(unique_vals)-1]
                ))
            else:
                # Numerical parameter
                dimensions.append(dict(
                    label=param,
                    values=df[param],
                    range=[df[param].min(), df[param].max()]
                ))
        
        # Create plot
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df['F1_Score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="F1 Score", x=1.02)
            ),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            font=dict(size=11),
            height=500,
            margin=dict(l=50, r=120, t=50, b=50)
        )
        
        return fig
    
    def run_optimization(self, n_trials: int = 100, wandb_config: Dict[str, Any] = None) -> optuna.Study:
        """Run HPO optimization with W&B tracking"""
        # Initialize W&B if config provided
        if wandb_config:
            wandb_settings = self.base_config.get('wandb_settings', {})
            project = wandb_settings.get('project', 'HPO')
            group = wandb_settings.get('group', self.study_name)
            tags = wandb_settings.get('tags', []) + [f'stage_{self.stage}']
            
            self.wandb_run = wandb.init(
                project=project,
                group=group,
                name=f"{self.study_name}_stage_{self.stage}",
                job_type=f"hpo_stage_{self.stage}",
                tags=tags,
                config={
                    **self.base_config,
                    'stage': self.stage,
                    'n_trials': n_trials,
                    'search_space': self.search_space
                }
            )
        
        logger.info("="*60)
        logger.info(f"TWO-STAGE HYPERPARAMETER OPTIMIZATION - STAGE {self.stage}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Results will be saved to: {self.results_dir}")
        logger.info(f"Search space parameters: {list(self.search_space.keys())}")
        logger.info("="*60)
        
        # Load Stage 1 results if this is Stage 2
        if self.stage == 2:
            stage1_results = self._load_stage1_results()
            if stage1_results:
                logger.info("Loading Stage 1 recommendations...")
                logger.info(f"Best Stage 1 F1 Score: {stage1_results.get('best_f1_score', 'N/A')}")
            else:
                logger.warning("No Stage 1 results found. Running Stage 2 with full search space.")
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{self.study_name}_stage_{self.stage}",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=8,
                n_warmup_steps=10,
                interval_steps=3
            ),
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=15,
                multivariate=True,
                warn_independent_sampling=False
            )
        )
        
        logger.info(f"Starting optimization with {n_trials} trials")
        
        # Add callback for W&B logging
        def trial_callback(study, trial):
            if self.wandb_run and trial.state == optuna.trial.TrialState.COMPLETE:
                wandb.log({
                    f"stage_{self.stage}/completed_trials": len([t for t in study.trials 
                                                                if t.state == optuna.trial.TrialState.COMPLETE]),
                    f"stage_{self.stage}/best_f1_so_far": 1 - study.best_value,
                })
        
        study.optimize(self.objective, n_trials=n_trials, callbacks=[trial_callback])
        
        # Analyze and save results
        self._analyze_and_save_results(study)
        
        # Close W&B run
        if self.wandb_run:
            wandb.finish()
        
        return study
    
    def _extract_stage2_recommendations(self, trials: List[optuna.Trial]) -> Dict[str, Any]:
        """Extract parameter recommendations for Stage 2 from top trials"""
        # Get top 20% trials
        sorted_trials = sorted(trials, key=lambda t: t.value)
        top_trials = sorted_trials[:max(1, len(sorted_trials) // 5)]
        
        recommendations = {
            'best_f1_score': 1 - sorted_trials[0].value,
            'best_trial_number': sorted_trials[0].number,
            'parameter_ranges': {}
        }
        
        # Collect parameter values from top trials
        param_values = {}
        for trial in top_trials:
            for param, value in trial.params.items():
                if param not in param_values:
                    param_values[param] = []
                param_values[param].append(value)
        
        # Create narrowed ranges
        for param, values in param_values.items():
            if all(isinstance(v, (int, float)) for v in values):
                min_val = min(values)
                max_val = max(values)
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # For numerical parameters, suggest narrowed range
                if isinstance(min_val, float):
                    # For log-scale parameters (learning rate, weight decay)
                    if param in ['learning_rate', 'weight_decay']:
                        # Use geometric mean and expand by factor of 3
                        geo_mean = np.exp(np.mean(np.log(values)))
                        new_min = geo_mean / 3
                        new_max = geo_mean * 3
                    else:
                        # For linear parameters, use mean ± 1.5 std
                        new_min = max(0, mean_val - 1.5 * std_val)
                        new_max = mean_val + 1.5 * std_val
                    
                    recommendations['parameter_ranges'][param] = {
                        'min': new_min,
                        'max': new_max,
                        'best': values[0],
                        'mean': mean_val,
                        'std': std_val
                    }
                else:
                    # Integer parameters
                    range_size = max_val - min_val
                    new_min = max(1, min_val - range_size // 2)
                    new_max = max_val + range_size // 2
                    
                    recommendations['parameter_ranges'][param] = {
                        'min': int(new_min),
                        'max': int(new_max),
                        'best': int(values[0])
                    }
            else:
                # Categorical parameters - keep most frequent values
                from collections import Counter
                value_counts = Counter(values)
                top_values = [v for v, _ in value_counts.most_common(3)]
                
                recommendations['parameter_ranges'][param] = {
                    'choices': top_values,
                    'best': values[0],
                    'frequency': dict(value_counts)
                }
        
        return recommendations
    
    def _analyze_and_save_results(self, study: optuna.Study):
        """Analyze results and save to files"""
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            logger.error("No completed trials!")
            return
        
        # Calculate statistics
        f1_scores = [1 - t.value for t in completed_trials]
        best_f1 = max(f1_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        # Create results summary
        summary_lines = [
            "="*60,
            f"HYPERPARAMETER OPTIMIZATION RESULTS - STAGE {self.stage}",
            "="*60,
            f"Model: {self.model_name}",
            f"Study: {self.study_name}",
            f"Stage: {self.stage}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Trials: {len(study.trials)}",
            f"Completed Trials: {len(completed_trials)}",
            f"Pruned Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}",
            "",
            "PERFORMANCE SUMMARY:",
            "-" * 30,
            f"Best F1 Score: {best_f1:.4f}",
            f"Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}",
            f"Best Trial: #{study.best_trial.number}",
            "",
            "BEST HYPERPARAMETERS:",
            "-" * 30,
        ]
        
        # Add best parameters
        for param, value in sorted(study.best_params.items()):
            summary_lines.append(f"  {param}: {value}")
        
        # Add parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            summary_lines.extend([
                "",
                "PARAMETER IMPORTANCE:",
                "-" * 30,
            ])
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                summary_lines.append(f"  {param}: {imp:.3f}")
        except:
            summary_lines.append("\nParameter importance analysis not available")
        
        # Add stage-specific recommendations
        if self.stage == 1:
            # Extract recommendations for Stage 2
            recommendations = self._extract_stage2_recommendations(completed_trials)
            
            summary_lines.extend([
                "",
                "STAGE 2 RECOMMENDATIONS:",
                "-" * 30,
                f"Based on top {len(completed_trials) // 5} trials from Stage 1:",
                "",
            ])
            
            for param, info in recommendations['parameter_ranges'].items():
                if 'choices' in info:
                    summary_lines.append(f"  {param}: Keep values {info['choices']} (best: {info['best']})")
                else:
                    if 'mean' in info:
                        summary_lines.append(
                            f"  {param}: [{info['min']:.2e}, {info['max']:.2e}] "
                            f"(best: {info['best']:.2e}, mean: {info['mean']:.2e})"
                        )
                    else:
                        summary_lines.append(
                            f"  {param}: [{info['min']}, {info['max']}] (best: {info['best']})"
                        )
            
            # Save recommendations for Stage 2
            recommendations_file = self.results_dir / "stage2_recommendations.json"
            with open(recommendations_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                def convert_to_serializable(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(v) for v in obj]
                    return obj
                
                json.dump(convert_to_serializable(recommendations), f, indent=2)
            
            summary_lines.extend([
                "",
                "To run Stage 2 with these recommendations:",
                f"1. Edit the config file to uncomment and modify stage_2_search_space",
                f"2. Run: python hpo.py --config {self.base_config_path} --stage 2 --trials 100",
                "",
                f"Stage 2 recommendations saved to: {recommendations_file}",
            ])
        
        elif self.stage == 2:
            # Compare with Stage 1 results if available
            stage1_results = self._load_stage1_results()
            if stage1_results:
                stage1_best_f1 = stage1_results.get('best_f1_score', 0)
                improvement = ((best_f1 - stage1_best_f1) / stage1_best_f1) * 100
                
                summary_lines.extend([
                    "",
                    "COMPARISON WITH STAGE 1:",
                    "-" * 30,
                    f"Stage 1 Best F1: {stage1_best_f1:.4f}",
                    f"Stage 2 Best F1: {best_f1:.4f}",
                    f"Improvement: {improvement:.2f}%",
                ])
        
        summary_lines.extend([
            "",
            "="*60,
        ])
        
        # Save text summary
        summary_file = self.results_dir / "hpo_results.txt"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        # Save best configuration
        best_config = self.base_config.copy()
        # Remove search spaces
        for key in list(best_config.keys()):
            if key.endswith('_search_space'):
                del best_config[key]
        
        best_config.update(study.best_params)
        best_config['_hpo_metadata'] = {
            'best_f1_score': best_f1,
            'best_trial': study.best_trial.number,
            'total_trials': len(study.trials),
            'study_name': self.study_name,
            'stage': self.stage,
            'timestamp': datetime.now().isoformat()
        }
        
        config_file = self.results_dir / f"best_config_stage_{self.stage}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        
        # Create and save interactive plot
        fig = self.create_parallel_coordinates_plot(
            study, f"HPO Results: {self.model_name} - Stage {self.stage}"
        )
        if fig:
            plot_file = self.results_dir / f"parallel_coordinates_stage_{self.stage}.html"
            fig.write_html(plot_file)
            logger.info(f"Interactive plot saved: {plot_file}")
        
        # Save study for later analysis
        study_file = self.results_dir / f"optuna_study_stage_{self.stage}.pkl"
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        
        # Log results
        logger.info("\n" + '\n'.join(summary_lines))
        logger.info(f"\nFiles saved in: {self.results_dir}")
        logger.info(f"  📄 Results summary: {summary_file}")
        logger.info(f"  ⚙️  Best config: {config_file}")
        if fig:
            logger.info(f"  📊 Interactive plot: {plot_file}")
        logger.info(f"  💾 Study data: {study_file}")
        
        logger.info(f"\n🚀 To train with best config:")
        logger.info(f"   python main.py --config {config_file} --wandb")


def main():
    parser = argparse.ArgumentParser(description="Two-Stage HPO with W&B Tracking")
    parser.add_argument("--config", type=str, required=True, help="Path to base config YAML")
    parser.add_argument("--study_name", type=str, default="two_stage_hpo", help="Study name")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1, 
                       help="HPO stage (1: broad exploration, 2: narrow refinement)")
    parser.add_argument("--trials", type=int, default=150, help="Number of trials")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B tracking")
    
    args = parser.parse_args()
    
    # Run optimization
    tuner = TwoStageHPOTuner(args.config, args.study_name, args.stage)
    
    # Set up W&B config if enabled
    wandb_config = {'enabled': True} if args.wandb else None
    
    study = tuner.run_optimization(args.trials, wandb_config)
    
    logger.info("\n" + "="*60)
    logger.info("HPO COMPLETED!")
    logger.info("="*60)


if __name__ == "__main__":
    main()