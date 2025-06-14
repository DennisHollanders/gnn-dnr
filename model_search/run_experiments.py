# experiment_runner.py
import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import logging
import traceback

# Add your project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class ExperimentConfig:
    """Configuration for a single experiment"""
    def __init__(self, model_name, method, variant, warmstart_params=None):
        self.model_name = model_name
        self.method = method
        self.variant = variant
        self.warmstart_params = warmstart_params or {}
        
    def to_dict(self):
        return {
            'model_name': self.model_name,
            'method': self.method,
            'variant': self.variant,
            'warmstart_params': self.warmstart_params
        }
    
    def get_job_name(self):
        """Generate unique job name for this configuration"""
        if self.method == "DirectPrediction":
            return f"{self.model_name}_{self.method}_{self.variant}"
        elif self.method == "HardWarmStart":
            threshold = self.warmstart_params.get('confidence_threshold', 0.5)
            return f"{self.model_name}_{self.method}_thresh{threshold}"
        else:
            return f"{self.model_name}_{self.method}_{self.variant}"

def generate_experiment_configs(model_names=['GIN']):
    """Generate all experiment configurations"""
    configs = []
    
    for model_name in model_names:
        try:
            # Direct Prediction methods (no optimization)
            configs.extend([
                ExperimentConfig(model_name, "DirectPrediction", "Rounding"),
                ExperimentConfig(model_name, "DirectPrediction", "PhyR")
            ])
            
            # Soft WarmStart methods
            configs.extend([
                ExperimentConfig(model_name, "SoftWarmStart", "Floats", 
                               {'warmstart_mode': 'float'}),
                ExperimentConfig(model_name, "SoftWarmStart", "BinaryRounding", 
                               {'warmstart_mode': 'soft', 'use_rounding': True}),
                ExperimentConfig(model_name, "SoftWarmStart", "BinaryPhyR", 
                               {'warmstart_mode': 'soft', 'use_phyr': True})
            ])
            
            # Hard WarmStart with different thresholds
            for threshold in [0.9, 0.7, 0.5, 0.3, 0.1]:
                configs.append(
                    ExperimentConfig(model_name, "HardWarmStart", str(threshold),
                                   {'warmstart_mode': 'hard', 'confidence_threshold': threshold})
                )
        except Exception as e:
            logging.error(f"Error generating configs for model {model_name}: {e}")
            continue
    
    return configs

def safe_load_networks(test_folder, logger):
    """Safely load networks with error handling"""
    try:
        from load_data import load_pp_networks
        pp_networks = load_pp_networks(test_folder)
        
        if "mst" not in pp_networks:
            logger.error("No 'mst' networks found in test folder")
            return None
            
        return pp_networks["mst"]
    except Exception as e:
        logger.error(f"Failed to load networks from {test_folder}: {e}")
        logger.error(traceback.format_exc())
        return None

def safe_load_predictions(predictions_path, logger):
    """Safely load predictions with error handling"""
    try:
        with open(predictions_path, 'rb') as f:
            predictions = pickle.load(f)
        logger.info(f"Loaded predictions for {len(predictions)} graphs")
        return predictions
    except FileNotFoundError:
        logger.error(f"Predictions file not found: {predictions_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        logger.error(traceback.format_exc())
        return None

def run_direct_prediction(config, predictions_path, test_networks, output_dir, logger):
    """Run direct prediction methods with comprehensive error handling"""
    logger.info(f"Running {config.method} - {config.variant}")
    
    results = []
    failed_graphs = []
    
    # Load predictions
    predictions = safe_load_predictions(predictions_path, logger)
    if predictions is None:
        return pd.DataFrame()
    
    for graph_id, net in test_networks.items():
        try:
            if graph_id not in predictions:
                logger.warning(f"No predictions found for graph {graph_id}")
                continue
            
            switch_scores = predictions[graph_id]
            net_copy = net.deepcopy()
            
            # Apply the prediction method
            try:
                if config.variant == "Rounding":
                    # Simple rounding
                    for i, score in enumerate(switch_scores):
                        if i < len(net_copy.switch):
                            net_copy.switch.at[i, 'closed'] = bool(score > 0.5)
                
                elif config.variant == "PhyR":
                    # Physics-informed rounding
                    try:
                        from model_search.AdvancedMLP.AdvancedMLP import physics_informed_rounding
                        switch_states = physics_informed_rounding(
                            net_copy, switch_scores, 
                            check_radiality=True,
                            check_connectivity=True
                        )
                        for i, state in enumerate(switch_states):
                            if i < len(net_copy.switch):
                                net_copy.switch.at[i, 'closed'] = bool(state)
                    except ImportError:
                        logger.error("Could not import physics_informed_rounding")
                        # Fallback to simple rounding
                        for i, score in enumerate(switch_scores):
                            if i < len(net_copy.switch):
                                net_copy.switch.at[i, 'closed'] = bool(score > 0.5)
            except Exception as e:
                logger.error(f"Error applying {config.variant} to graph {graph_id}: {e}")
                failed_graphs.append(graph_id)
                continue
            
            # Evaluate the result
            try:
                result = evaluate_network(net_copy, net, graph_id, logger)
                result['method'] = config.method
                result['variant'] = config.variant
                result['optimization_time'] = 0  # No optimization
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating graph {graph_id}: {e}")
                failed_graphs.append(graph_id)
                
        except Exception as e:
            logger.error(f"Unexpected error processing graph {graph_id}: {e}")
            logger.error(traceback.format_exc())
            failed_graphs.append(graph_id)
    
    logger.info(f"Completed {len(results)} graphs, failed {len(failed_graphs)}")
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()

def run_warmstart_optimization(config, predictions_path, test_folder, output_dir, logger):
    """Run optimization with warmstart and comprehensive error handling"""
    logger.info(f"Running {config.method} - {config.variant}")
    
    try:
        # Import required modules
        from predict_then_optimize import Optimizer
        
        # Initialize optimizer
        try:
            optimizer = Optimizer(
                folder_name=str(test_folder),
                warmstart_path=str(predictions_path),
                results_folder=str(output_dir),
                warmstart_mode=config.warmstart_params['warmstart_mode'],
                confidence_threshold=config.warmstart_params.get('confidence_threshold', 0.5)
            )
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
        
        # Handle special preprocessing for soft warmstart variants
        if config.method == "SoftWarmStart":
            try:
                if config.variant == "BinaryRounding":
                    # Round predictions before using as warmstart
                    for graph_id in optimizer.warmstarts:
                        optimizer.warmstarts[graph_id] = (optimizer.warmstarts[graph_id] > 0.5).astype(float)
                
                elif config.variant == "BinaryPhyR":
                    # Apply PhyR before optimization
                    try:
                        from model_search.AdvancedMLP.AdvancedMLP import physics_informed_rounding
                        for graph_id in optimizer.warmstarts:
                            try:
                                net = optimizer.pp_all["mst"][graph_id]
                                phyr_states = physics_informed_rounding(
                                    net, optimizer.warmstarts[graph_id],
                                    check_radiality=True,
                                    check_connectivity=True
                                )
                                optimizer.warmstarts[graph_id] = phyr_states.astype(float)
                            except Exception as e:
                                logger.warning(f"PhyR failed for graph {graph_id}, using simple rounding: {e}")
                                optimizer.warmstarts[graph_id] = (optimizer.warmstarts[graph_id] > 0.5).astype(float)
                    except ImportError:
                        logger.error("Could not import physics_informed_rounding, falling back to simple rounding")
                        for graph_id in optimizer.warmstarts:
                            optimizer.warmstarts[graph_id] = (optimizer.warmstarts[graph_id] > 0.5).astype(float)
            except Exception as e:
                logger.error(f"Error preprocessing warmstarts: {e}")
                logger.error(traceback.format_exc())
        
        # Run optimization
        try:
            results = optimizer.run(num_workers=1)
            return process_optimization_results(results, config, logger)
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Unexpected error in warmstart optimization: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def evaluate_network(net_result, net_original, graph_id, logger):
    """Evaluate a network configuration with error handling"""
    result = {
        'graph_id': graph_id,
        'radial': False,
        'connected': False,
        'feasible': False,
        'pf_converged': False,
        'switches_changed': 0,
        'losses_mw': np.nan
    }
    
    try:
        from define_ground_truth import is_radial_and_connected, count_switch_changes
        import pandapower as pp
        
        # Check radiality and connectivity
        try:
            radial, connected = is_radial_and_connected(net_result, include_switches=True)
            result['radial'] = radial
            result['connected'] = connected
        except Exception as e:
            logger.warning(f"Failed to check radiality/connectivity for {graph_id}: {e}")
        
        # Try to run power flow
        try:
            pp.runpp(net_result, enforce_q_lims=False)
            result['pf_converged'] = net_result.converged
            result['feasible'] = result['pf_converged'] and result['radial'] and result['connected']
            
            # Calculate losses if power flow converged
            if result['pf_converged']:
                result['losses_mw'] = net_result.res_line["pl_mw"].sum()
        except Exception as e:
            logger.warning(f"Power flow failed for {graph_id}: {e}")
        
        # Count switch changes
        try:
            result['switches_changed'] = count_switch_changes(net_original, net_result)
        except Exception as e:
            logger.warning(f"Failed to count switch changes for {graph_id}: {e}")
            
    except Exception as e:
        logger.error(f"Unexpected error evaluating network {graph_id}: {e}")
        logger.error(traceback.format_exc())
    
    return result

def process_optimization_results(results_dict, config, logger):
    """Process optimization results into DataFrame format"""
    processed_results = []
    
    for graph_id, result in results_dict.items():
        try:
            processed = {
                'graph_id': graph_id,
                'method': config.method,
                'variant': config.variant,
                'optimization_time': result.get('solve_time', np.nan),
                'feasible': result.get('success', False),
                'radial': result.get('after_radial', False),
                'connected': result.get('after_connected', False),
                'switches_changed': result.get('switches_changed', 0),
                'losses_mw': result.get('objective', np.nan),
                'pf_converged': result.get('pf_converged', False)
            }
            processed_results.append(processed)
        except Exception as e:
            logger.error(f"Error processing result for graph {graph_id}: {e}")
            continue
    
    return pd.DataFrame(processed_results)

def run_single_experiment(config, model_path, test_folder, output_base_dir, logger):
    """Run a single experiment with full error handling"""
    output_dir = Path(output_base_dir) / config.get_job_name()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load test networks
        test_networks = safe_load_networks(test_folder, logger)
        if test_networks is None:
            return None
        
        # Check for predictions
        predictions_path = Path(output_base_dir) / "predictions.pkl"
        
        if config.method == "DirectPrediction":
            return run_direct_prediction(
                config, predictions_path, test_networks, output_dir, logger
            )
        else:
            return run_warmstart_optimization(
                config, predictions_path, test_folder, output_dir, logger
            )
            
    except Exception as e:
        logger.error(f"Failed to run experiment {config.get_job_name()}: {e}")
        logger.error(traceback.format_exc())
        return None

def generate_summary_table(results_df, output_path, logger):
    """Generate summary table with error handling"""
    try:
        # Calculate metrics for each configuration
        summary = results_df.groupby(['method', 'variant']).agg({
            'optimization_time': 'mean',
            'feasible': lambda x: (~x).sum(),  # Count infeasible
            'radial': lambda x: (~x).sum(),    # Count non-radial
            'connected': lambda x: (~x).sum(),  # Count disconnected
            'switches_changed': 'mean',
            'losses_mw': 'mean'
        }).round(2)
        
        summary.to_csv(output_path)
        logger.info(f"Summary table saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate summary table: {e}")
        logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to config file")
    parser.add_argument("--config_name", type=str, help="Specific config to run")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiment_results")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting experiment: {args.config_name}")
    
    try:
        # Load configuration
        if args.config_file:
            with open(args.config_file, 'r') as f:
                config_dict = json.load(f)
            config = ExperimentConfig(**config_dict)
        else:
            # Generate config based on name
            parts = args.config_name.split('_')
            if len(parts) >= 3:
                model_name = parts[0]
                method = parts[1]
                variant = '_'.join(parts[2:])
                
                # Reconstruct config
                if method == "DirectPrediction":
                    config = ExperimentConfig(model_name, method, variant)
                elif method == "HardWarmStart":
                    threshold = float(variant.replace('thresh', ''))
                    config = ExperimentConfig(
                        model_name, method, str(threshold),
                        {'warmstart_mode': 'hard', 'confidence_threshold': threshold}
                    )
                else:
                    # Handle other cases
                    config = ExperimentConfig(model_name, method, variant)
        
        # Run experiment
        result = run_single_experiment(
            config, args.model_path, args.test_folder, args.output_dir, logger
        )
        
        if result is not None and not result.empty:
            # Save results
            output_path = Path(args.output_dir) / config.get_job_name() / "results.csv"
            result.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        else:
            logger.error("No results generated")
            
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()