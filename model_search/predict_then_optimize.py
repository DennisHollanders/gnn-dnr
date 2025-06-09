# predict_optimize.py
import sys
import os
import torch
import torch.multiprocessing as mp
from torch_geometric.data import DataLoader
import pandapower as pp
import pickle
import yaml
import importlib
from pathlib import Path
import math
from tqdm import tqdm
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cvxpy_SOCP import build_misocp_problem
from src.SOCP_class_dnr import SOCP_class
from model_search.evaluation.evaluation import load_config_from_model_path

class Predictor:
    def __init__(self,model_path: str,config_path: str,device: torch.device,sample_loader: DataLoader):
        self.device = device

    
        if config_path:
            print(f"Loading configuration from: {config_path}")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            print(f"Auto-detecting config next to: {os.path.dirname(model_path)}")
            config = load_config_from_model_path(model_path)


        self.eval_args = argparse.Namespace(**config)

        print(f"\nLoaded configuration:")
        print(f"  Model module: {self.eval_args.model_module}")
        print(f"  Hidden dims: {self.eval_args.hidden_dims}")
        print(f"  Latent dim: {self.eval_args.latent_dim}")
        print(f"  Activation: {self.eval_args.activation}")
        print(f"  Dropout rate: {self.eval_args.dropout_rate}")
        print(f"  Job name: {self.eval_args.job_name}\n")


        sample_data = sample_loader.dataset[0]
        node_input_dim = sample_data.x.shape[1]
        edge_input_dim = sample_data.edge_attr.shape[1]

        print(f"Data dimensions inferred:")
        print(f"  Node input dim: {node_input_dim}")
        print(f"  Edge input dim: {edge_input_dim}\n")

        model_module = importlib.import_module(f"models.{self.eval_args.model_module}.{self.eval_args.model_module}")
        model_class = getattr(model_module, self.eval_args.model_module)

        self.model = model_class(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dims=self.eval_args.hidden_dims,
            latent_dim=self.eval_args.latent_dim,
            activation=self.eval_args.activation,
            dropout_rate=self.eval_args.dropout_rate,
        ).to(self.device)

        # 4) Load pretrained weights
        print(f"Loading model weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            self.model.load_state_dict(state_dict["model_state_dict"])
        else:
            self.model.load_state_dict(state_dict)
        print("Model loaded successfully!")

        self.model.eval()

    def run(self, dataloader: DataLoader, warmstart_path: str):
        warmstarts = {}
        sample_idx = 0
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Predicting switch scores", unit="batch", leave=False):

                data = data.to(self.device)
                output = self.model(data)
                scores = output.get("switch_scores")
                if scores is None:
                    raise RuntimeError("Model output must contain 'switch_scores'")
                scores = scores.detach().cpu().squeeze(-1).numpy()
                warmstarts[sample_idx] = scores
                sample_idx += 1

        os.makedirs(os.path.dirname(warmstart_path), exist_ok=True)
        with open(warmstart_path, "wb") as f:
            pickle.dump(warmstarts, f)


def apply_hard_warmstart(net, switch_scores, confidence_threshold=0.8):
    """
    Modify the network to fix high-confidence switches by setting them as non-switchable.
    Returns a modified network and a list of lines to fix in the optimizer.
    """
    net_modified = net.deepcopy()
    fixed_lines = {}  # line_idx: desired_status
    
    print(f"Applying hard warmstart with confidence threshold {confidence_threshold}")
    
    # Identify high-confidence switches
    for i, score in enumerate(switch_scores):
        confidence = abs(score - 0.5) * 2  # Distance from 0.5, scaled to [0,1]
        
        if confidence >= confidence_threshold:
            desired_status = 1 if score >= 0.5 else 0
            
            # Find corresponding switch in the network
            if i < len(net_modified.switch):
                switch_row = net_modified.switch.iloc[i]
                if switch_row.et == 'l':  # Line switch
                    line_idx = switch_row.element
                    fixed_lines[line_idx] = desired_status
                    
                    # Set the switch status in the network
                    net_modified.switch.at[i, 'closed'] = bool(desired_status)
                    
                    print(f"Fixed line {line_idx} to {'ON' if desired_status else 'OFF'} (score: {score:.3f}, confidence: {confidence:.3f})")
    
    print(f"Hard warmstart: Fixed {len(fixed_lines)} lines")
    return net_modified, fixed_lines


def apply_soft_warmstart(net, switch_scores):
    """
    Modify the network switch configuration based on predictions.
    """
    net_modified = net.deepcopy()
    
    print("Applying soft warmstart by modifying network switch configuration")
    
    # Set switch states based on predictions
    for i, score in enumerate(switch_scores):
        if i < len(net_modified.switch):
            switch_status = 1 if score >= 0.5 else 0
            net_modified.switch.at[i, 'closed'] = bool(switch_status)
            print(f"Switch {i}: set to {'ON' if switch_status else 'OFF'} (score: {score:.3f})")
    
    return net_modified


def solve_with_float_warmstart(optimizer, switch_scores, **solver_kwargs):
    """
    Custom solve method that applies float values to binary variables.
    """
    print("Solving with float warmstart...")
    
    # Create model if not exists
    if optimizer.model is None:
        optimizer.create_model()
    
    m = optimizer.model
    
    # Apply float warmstart values
    if optimizer.toggles.get('all_lines_are_switches', False):
        # Apply to line_status variables
        line_idx = 0
        for l in m.lines:
            if line_idx < len(switch_scores) and l in optimizer.lines_with_sw:
                float_value = max(0.0, min(1.0, float(switch_scores[line_idx])))
                m.line_status[l].set_value(float_value)
                print(f"Float-initialized line {l} to {float_value:.3f}")
                line_idx += 1
    else:
        # Apply to switch_status variables if they exist
        if hasattr(m, 'model_switches'):
            switch_idx = 0
            for s in m.model_switches:
                if switch_idx < len(switch_scores):
                    float_value = max(0.0, min(1.0, float(switch_scores[switch_idx])))
                    m.switch_status[s].set_value(float_value)
                    print(f"Float-initialized switch {s} to {float_value:.3f}")
                    switch_idx += 1
    
    # Solve normally
    return optimizer.solve(**solver_kwargs)


class Optimizer:
    """
    Compact optimizer that uses minimal modifications to the existing SOCP_class.
    """
    
    def __init__(self, 
                 folder_name: str,
                 warmstart_path: str,
                 results_folder: str,
                 warmstart_mode: str = "none",
                 confidence_threshold: float = 0.8):
        
        self.folder_name = folder_name
        self.results_folder = results_folder
        self.warmstart_mode = warmstart_mode
        self.confidence_threshold = confidence_threshold
        
        os.makedirs(self.results_folder, exist_ok=True)

        # Load warmstart data
        with open(warmstart_path, "rb") as f:
            self.warmstarts = pickle.load(f)

        from load_data import load_pp_networks
        pp_all = load_pp_networks(self.folder_name)
        self.graph_ids = sorted(pp_all["mst"].keys())

        if len(self.graph_ids) != len(self.warmstarts):
            raise ValueError(
                "Number of warmstarts does not match number of networks in folder_name"
            )

        self.pp_all = pp_all

    def _create_modified_socp_class(self, fixed_lines=None):
        """
        Create a modified SOCP_class that fixes certain lines.
        """
        class FixedLineSOCP(SOCP_class):
            def __init__(self, *args, fixed_lines_dict=None, **kwargs):
                self.fixed_lines_dict = fixed_lines_dict or {}
                super().__init__(*args, **kwargs)
            
            def create_model(self):
                # Call parent create_model
                model = super().create_model()
                
                # Fix additional lines based on warmstart
                if self.fixed_lines_dict:
                    for line_idx, desired_status in self.fixed_lines_dict.items():
                        if line_idx in model.lines:
                            model.line_status[line_idx].fix(desired_status)
                            print(f"Fixed line {line_idx} to {desired_status} (warmstart)")
                
                return model
        
        return FixedLineSOCP

    def _optimize_single(self, args):
        idx, mode = args
        gid = self.graph_ids[idx]

        # Load original pandapower network
        net = self.pp_all["mst"][gid]
        switch_scores = self.warmstarts[idx]
        
        toggles = {
            "include_voltage_drop_constraint": True,
            "include_voltage_bounds_constraint": True,
            "include_power_balance_constraint": True,
            "include_radiality_constraints": True,
            "use_root_flow": True,
            "include_switch_penalty": True,
            "allow_load_shed": False,
            "include_cone_constraint": True,
        }

        try:
            if mode == "hard":
                # Hard warmstart: fix high-confidence switches
                net_modified, fixed_lines = apply_hard_warmstart(
                    net, switch_scores, self.confidence_threshold
                )
                
                # Use modified SOCP class that respects fixed lines
                FixedSOCP = self._create_modified_socp_class(fixed_lines)
                optimizer = FixedSOCP(
                    net=net_modified,
                    toggles=toggles,
                    graph_id=gid
                )
                optimizer.initialize()
                results = optimizer.solve(solver="gurobi", time_limit=300, mip_gap=1e-3)
                
            elif mode == "soft":
                # Soft warmstart: modify network switch configuration
                net_modified = apply_soft_warmstart(net, switch_scores)
                
                optimizer = SOCP_class(
                    net=net_modified,
                    toggles=toggles,
                    graph_id=gid
                )
                optimizer.initialize()
                results = optimizer.solve(solver="gurobi", time_limit=300, mip_gap=1e-3)
                
            elif mode == "float_soft":
                # Float warmstart: use custom solve method
                optimizer = SOCP_class(
                    net=net,
                    toggles=toggles,
                    graph_id=gid
                )
                optimizer.initialize()
                results = solve_with_float_warmstart(
                    optimizer, switch_scores,
                    solver="gurobi", time_limit=300, mip_gap=1e-3
                )
                
            else:  # mode == "none"
                # No warmstart: use original network
                optimizer = SOCP_class(
                    net=net,
                    toggles=toggles,
                    graph_id=gid
                )
                optimizer.initialize()
                results = optimizer.solve(solver="gurobi", time_limit=300, mip_gap=1e-3)
            
            # Extract solution
            if optimizer.model and hasattr(optimizer.model, 'line_status'):
                solution = {}
                for l in optimizer.model.lines:
                    solution[l] = round(pyo_val(optimizer.model.line_status[l]))
                    
                return gid, {
                    'solution': solution,
                    'objective': pyo_val(optimizer.model.objective),
                    'solve_time': optimizer.solve_time,
                    'switches_changed': optimizer.num_switches_changed,
                    'warmstart_mode': mode,
                    'success': True
                }
            else:
                return gid, {
                    'solution': None,
                    'success': False,
                    'error': 'No valid solution found',
                    'warmstart_mode': mode
                }
                
        except Exception as e:
            print(f"Error optimizing network {gid}: {str(e)}")
            return gid, {
                'solution': None,
                'success': False,
                'error': str(e),
                'warmstart_mode': mode
            }

    def run(self, num_workers: int = None):
        """Run optimization with warmstart support."""
        if num_workers is None:
            num_workers = max(1, os.cpu_count() - 1)
            
        print(f"Running optimization with '{self.warmstart_mode}' warmstart using {num_workers} workers")
        
        args_iterable = [(i, self.warmstart_mode) for i in range(len(self.graph_ids))]
        
        # Use multiprocessing or serial processing
        if num_workers > 1:
            with mp.Pool(processes=num_workers) as pool:
                it = pool.imap_unordered(self._optimize_single, args_iterable)
                results = {}
                for gid, result in tqdm(it, 
                                      total=len(self.graph_ids),
                                      desc=f"Optimizing with {self.warmstart_mode} warmstart",
                                      unit="network"):
                    results[gid] = result
        else:
            # Serial processing for debugging
            results = {}
            for idx in tqdm(range(len(self.graph_ids)), 
                           desc=f"Optimizing with {self.warmstart_mode} warmstart", 
                           unit="network"):
                gid, result = self._optimize_single((idx, self.warmstart_mode))
                results[gid] = result

        return results


if __name__ == "__main__":
    import argparse
    import numpy as np
    import cvxpy as cp

    parser = argparse.ArgumentParser(description="Predict-then-Optimize Pipeline")
    parser.add_argument("--config_path",type=str,default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\model_search\models\MLP\config_files\MLP------clear-monkey-40.yaml",help="Path to the YAML config file. If not provided, will attempt to auto-detect.",)
    parser.add_argument("--model_path", type=str,default = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\model_search\models\MLP\clear-monkey-40-Best.pt", help="Path to pretrained GNN checkpoint")
    parser.add_argument("--folder_names",type=str,default = [r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\source_datasets\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32\test"],help="Folder containing 'mst' and 'mst_opt' subfolders",)
    parser.add_argument("--dataset_names",type=str,default = ["test"],help="Three folders for train/validation/test (each must have 'mst', 'mst_opt')",    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for Predictor (use 1 to simplify warmstart indexing)")
    parser.add_argument("--results_folder",type=str,default="predict_opt_results",help="Where to store optimization outputs",)
    parser.add_argument("--mode",type=str,choices=["warmstart", "threshold"],default="warmstart",help="Whether to run warmstart-based solve or threshold-fixed solve")
    parser.add_argument("--threshold",type=float,default=[0.5],help="Threshold for rounding in 'threshold' mode",)
    parser.add_argument("--num_workers",type=int,default=None,help="Number of CPU workers for optimization",)
    parser.add_argument("--predict", default=True, help="Run prediction step before optimization")
    parser.add_argument("--optimize", default=True, help="Run optimization step after prediction")
    args = parser.parse_args()

    # 1) Build DataLoaders for train/val/test using load_data.create_data_loaders
    from load_data import create_data_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_data_loaders(
        dataset_names= args.dataset_names,
        folder_names=args.folder_names,
        dataset_type="cvx",
        batch_size=args.batch_size,
        batching_type="standard",
    )
    test_loader = loaders.get("test", None)

    print(f"Train loader created with {len(test_loader.dataset)} samples")

    # --model_path", type=str,default = r"model_search\models\MLP\clear-monkey-40-Best.pt split jobname from model_path
    #job_name = str(os.path.basename(args.model_path).split(".")[0])
    # Extract job name from model_path
    job_name = Path(args.model_path).stem
    print(f"Job name extracted from model path: {job_name}")

    # Build a base directory under the first data folder to store all outputs
    data_folder_base = Path(args.folder_names[0])
    results_base = data_folder_base / args.results_folder / job_name
    results_base.mkdir(parents=True, exist_ok=True)

    # Construct full warmstart path (e.g., data/.../predict_opt_results/clear-monkey-40-Best/clear-monkey-40-Best_warmstarts.pkl)
    full_warmstart_path = (results_base / f"{job_name}_warmstarts.pkl").resolve()
    print(f"Warmstart path: {full_warmstart_path}")

    if args.predict:            
        print("Starting prediction...")
        # 2) Run Predictor on train/val/test (if desired) or just test set
        predictor = Predictor(
        model_path=args.model_path,
        config_path=args.config_path,
        device=device,
        sample_loader=test_loader)

        predictor.run(test_loader,str(full_warmstart_path) )

    if args.optimize:
        print("Starting optimization...")
        if len(args.threshold) > 1 and args.mode == "threshold":
            print(f"Multiple thresholds provided: {args.threshold}. Running optimization for each threshold.") 
            for thr in args.threshold:
                print(f"Running optimization with threshold: {thr}")
                # Update threshold in results_base
                results_base_thr = results_base / f"threshold_{thr}"
                results_base_thr.mkdir(parents=True, exist_ok=True)
                full_warmstart_path_thr = results_base_thr / f"{job_name}_warmstarts.pkl"
                full_warmstart_path_thr = full_warmstart_path_thr.resolve()
                
                # 3) Run Optimizer
                optimizer = Optimizer(
                    folder_name=args.folder_names[0],  # Use the first folder name for the network data
                    warmstart_path=str(full_warmstart_path_thr),
                    results_folder=str(results_base_thr),
                    threshold=thr,
                )
                final_results = optimizer.run(mode=args.mode, num_workers=args.num_workers)

                summary_path = results_base_thr / "summary_results.pkl"
                with open(summary_path, "wb") as f:
                    pickle.dump(final_results, f)
                print(f"Summary saved to: {summary_path}")
        else:

            # 3) Run Optimizer
            optimizer = Optimizer(
                folder_name =args.folder_names[0],  
                warmstart_path=(full_warmstart_path),
                results_folder=(results_base),
                threshold=args.threshold,
            )
            final_results = optimizer.run(mode=args.mode, num_workers=args.num_workers)

            summary_path = results_base / "summary_results.pkl"
            with open(summary_path, "wb") as f:
                pickle.dump(final_results, f)
            print(f"Summary saved to: {summary_path}")
