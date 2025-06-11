# Corrected predict_optimize.py with simplified warmstart approach
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
import argparse  # Missing import

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyomo.environ import value as pyo_val
from src.SOCP_class_dnr import SOCP_class
from model_search.evaluation.evaluation import load_config_from_model_path

class Predictor:
    def __init__(self, model_path: str, config_path: str, device: torch.device, sample_loader: DataLoader):
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


class WarmstartSOCP(SOCP_class):
    """Enhanced SOCP class with warmstart support"""
    
    def __init__(self, *args, fixed_switches=None, float_warmstart=None, **kwargs):
        self.fixed_switches = fixed_switches or {}  # switch_idx: desired_status
        self.float_warmstart = float_warmstart  # array of float values for warmstart
        super().__init__(*args, **kwargs)
    
    def initialize(self):
        """Override to handle fixed switches"""
        super().initialize()
        
        # Handle fixed switches by modifying the switch DataFrame
        if self.fixed_switches:
            print(f"Fixing {len(self.fixed_switches)} switches based on warmstart")
            for switch_idx, desired_status in self.fixed_switches.items():
                if switch_idx < len(self.switch_df):
                    # Set the switch to desired status and mark as non-controllable
                    self.switch_df.loc[switch_idx, 'closed'] = bool(desired_status)
                    print(f"Fixed switch {switch_idx} to {'ON' if desired_status else 'OFF'}")
    
    def create_model(self):
        """Override to apply warmstart and fix switches"""
        model = super().create_model()
        
        # Fix switches that were marked as fixed
        if self.fixed_switches:
            for switch_idx, desired_status in self.fixed_switches.items():
                if switch_idx < len(self.switch_df):
                    switch_row = self.switch_df.iloc[switch_idx]
                    if switch_row.et == 'l':  # Line switch
                        line_idx = switch_row.element
                        if line_idx in model.lines:
                            model.line_status[line_idx].fix(desired_status)
                            print(f"Fixed line {line_idx} status to {desired_status} in optimization model")
        
        return model
    
    def solve(self, *, solver: str = "gurobi_persistent", **solver_kw):
        """Override solve to apply float warmstart if provided"""
        if self.float_warmstart is not None:
            return self._solve_with_float_warmstart(solver=solver, **solver_kw)
        else:
            return super().solve(solver=solver, **solver_kw)
    
    def _solve_with_float_warmstart(self, *, solver: str = "gurobi_persistent", **solver_kw):
        """Solve with float warmstart values"""
        from pyomo.opt import SolverFactory
        import time
        
        if self.model is None:
            self.create_model()
        
        m = self.model
        
        # Apply float warmstart to binary variables
        warmstart_count = 0
        if self.toggles.get('all_lines_are_switches', False):
            # Apply to line_status variables
            switch_idx = 0
            for l in m.lines:
                if l in self.lines_with_sw and switch_idx < len(self.float_warmstart):
                    float_value = max(0.0, min(1.0, float(self.float_warmstart[switch_idx])))
                    m.line_status[l].set_value(float_value)
                    self.logger.debug(f"Float warmstart: line {l} = {float_value:.3f}")
                    switch_idx += 1
                    warmstart_count += 1
        else:
            # Apply to switch_status variables
            if hasattr(m, 'model_switches'):
                switch_idx = 0
                for s in m.model_switches:
                    if switch_idx < len(self.float_warmstart):
                        float_value = max(0.0, min(1.0, float(self.float_warmstart[switch_idx])))
                        m.switch_status[s].set_value(float_value)
                        self.logger.debug(f"Float warmstart: switch {s} = {float_value:.3f}")
                        switch_idx += 1
                        warmstart_count += 1
        
        print(f"Applied float warmstart to {warmstart_count} binary variables")
        
        # Solve with warmstart
        start = time.time()
        opt = SolverFactory(solver)
        if hasattr(opt, "set_instance"):
            opt.set_instance(m)
        
        # Set solver options including warmstart
        opt.options.update({
            "Threads": 8, 
            "TimeLimit": 300, 
            "MIPGap": 1e-3,
            "NonConvex": 2,
            "StartMethod": 1,  # Enable MIP start
        })
        opt.options.update(solver_kw)
        
        self.solver_results = opt.solve(tee=self.debug_level > 1, load_solutions=True)
        self.solve_time = time.time() - start
        
        obj_val = pyo_val(self.model.objective)
        self.logger.info("Solved with float warmstart in %.2fs (obj = %.3f)", self.solve_time, obj_val)
        
        return self.solver_results


class Optimizer:
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
                f"Number of warmstarts ({len(self.warmstarts)}) does not match number of networks ({len(self.graph_ids)})"
            )

        self.pp_all = pp_all

    def _optimize_single(self, args):
        idx, mode = args
        gid = self.graph_ids[idx]

        # Load original pandapower network
        net = self.pp_all["mst"][gid].deepcopy()  # Important: make a copy
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
            if mode == "soft": 
                # Soft warmstart: modify initial switch states
                for i, score in enumerate(switch_scores):
                    if i < len(net.switch):
                        net.switch.at[i, 'closed'] = bool(score > 0.5)
                
                optimizer = WarmstartSOCP(
                    net=net,
                    toggles=toggles,
                    graph_id=gid
                )
                
            elif mode == "float":
                # Float warmstart: provide float hints to solver
                optimizer = WarmstartSOCP(
                    net=net,
                    toggles=toggles,
                    graph_id=gid,
                    float_warmstart=switch_scores
                )
                
            elif mode == "hard":
                # Hard warmstart: fix high-confidence switches
                fixed_switches = {}
                for i, score in enumerate(switch_scores):
                    confidence = abs(score - 0.5) * 2  # Distance from 0.5, scaled to [0,1]
                    if confidence >= self.confidence_threshold:
                        fixed_switches[i] = 1 if score > 0.5 else 0
                        if i < len(net.switch):
                            net.switch.at[i, 'closed'] = bool(score > 0.5)
                
                optimizer = WarmstartSOCP(
                    net=net,
                    toggles=toggles,
                    graph_id=gid,
                    fixed_switches=fixed_switches
                )
                
            else:  # mode == "none"
                # No warmstart: use original network
                optimizer = SOCP_class(
                    net=net,
                    toggles=toggles,
                    graph_id=gid
                )

            # Initialize and solve
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
                    'switches_changed': getattr(optimizer, 'num_switches_changed', 0),
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
            import traceback
            traceback.print_exc()
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
        
        # Use serial processing for debugging, multiprocessing for production
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
    parser = argparse.ArgumentParser(description="Predict-then-Optimize Pipeline")
    parser.add_argument("--config_path", type=str,
                       default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\model_search\models\MLP\config_files\MLP------clear-monkey-40.yaml",
                       help="Path to the YAML config file")
    parser.add_argument("--model_path", type=str,
                       default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\model_search\models\MLP\clear-monkey-40-Best.pt", 
                       help="Path to pretrained GNN checkpoint")
    parser.add_argument("--folder_names", type=str,
                       default=[r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\source_datasets\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32\test"],
                       help="Folder containing 'mst' and 'mst_opt' subfolders")
    parser.add_argument("--dataset_names", type=str,
                       default=["test"],
                       help="Dataset names corresponding to folder_names")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Batch size for Predictor")
    parser.add_argument("--results_folder", type=str, default="predict_opt_results",
                       help="Where to store optimization outputs")
    
    # Warmstart arguments
    parser.add_argument("--warmstart_mode", type=str,
                       choices=["none", "soft", "float", "hard"],
                       default="none",
                       help="Warmstart strategy")
    parser.add_argument("--confidence_threshold", type=float, default=0.8,
                       help="Confidence threshold for hard warmstart")
    
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of CPU workers for optimization")
    parser.add_argument("--predict", action="store_true", default=True, 
                       help="Run prediction step")
    parser.add_argument("--optimize", action="store_true", default=True, 
                       help="Run optimization step")
    
    args = parser.parse_args()

    # Build DataLoaders
    from load_data import create_data_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_data_loaders(
        dataset_names=args.dataset_names,
        folder_names=args.folder_names,
        dataset_type="cvx",
        batch_size=args.batch_size,
        batching_type="standard",
    )
    test_loader = loaders.get("test", None)

    print(f"Test loader created with {len(test_loader.dataset)} samples")

    # Extract job name from model_path
    job_name = Path(args.model_path).stem
    print(f"Job name extracted from model path: {job_name}")

    # Build results directory
    data_folder_base = Path(args.folder_names[0])
    results_base = data_folder_base / args.results_folder / job_name / args.warmstart_mode
    if args.warmstart_mode == "hard":
        results_base = results_base / f"conf_{args.confidence_threshold}"
    results_base.mkdir(parents=True, exist_ok=True)

    # Construct warmstart path
    full_warmstart_path = (results_base / f"{job_name}_warmstarts.pkl").resolve()
    print(f"Warmstart path: {full_warmstart_path}")

    if args.predict:            
        print("Starting prediction...")
        predictor = Predictor(
            model_path=args.model_path,
            config_path=args.config_path,
            device=device,
            sample_loader=test_loader
        )
        predictor.run(test_loader, str(full_warmstart_path))

    if args.optimize:
        print(f"Starting optimization with '{args.warmstart_mode}' warmstart...")
        
        optimizer = Optimizer(
            folder_name=args.folder_names[0],  
            warmstart_path=str(full_warmstart_path),
            results_folder=str(results_base),
            warmstart_mode=args.warmstart_mode,
            confidence_threshold=args.confidence_threshold,
        )
        final_results = optimizer.run(num_workers=args.num_workers)

        # Save results
        summary_path = results_base / f"summary_results_{args.warmstart_mode}.pkl"
        with open(summary_path, "wb") as f:
            pickle.dump(final_results, f)
        print(f"Summary saved to: {summary_path}")
        
        # Print summary statistics
        successful = sum(1 for r in final_results.values() if r.get('success', False))
        total = len(final_results)
        if successful > 0:
            avg_solve_time = sum(r.get('solve_time', 0) for r in final_results.values() if r.get('success')) / successful
            avg_switches_changed = sum(r.get('switches_changed', 0) for r in final_results.values() if r.get('success')) / successful
        else:
            avg_solve_time = 0
            avg_switches_changed = 0
        
        print(f"\nOptimization Summary ({args.warmstart_mode} warmstart):")
        print(f"  Successful solves: {successful}/{total}")
        print(f"  Average solve time: {avg_solve_time:.2f}s")
        print(f"  Average switches changed: {avg_switches_changed:.1f}")
        if args.warmstart_mode == "hard":
            print(f"  Confidence threshold: {args.confidence_threshold}")