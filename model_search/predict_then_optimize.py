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


class Optimizer:
    def __init__(self,folder_name: str,warmstart_path: str,results_folder: str,threshold: float = 0.5,):
        self.folder_name = folder_name
        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)

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
        self.threshold = threshold

    def _optimize_single(self, args):
        idx, mode = args  
        gid = self.graph_ids[idx]

        # Load original pandapower network
        net = self.pp_all["mst"][gid]
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

        problem, variables, bus_map, line_list = build_misocp_problem(net, toggles, logger=None)

        y_var = variables["y_line"]
        warm_vals = self.warmstarts[idx] 

        if mode == "warmstart":
            y_var.value = warm_vals
            problem.solve(solver=cp.SCIPY, warm_start=True)  
        elif mode == "threshold":
            fixed = (warm_vals >= self.threshold).astype(float)
            for e_idx, val in enumerate(fixed):
                problem.constraints.append(y_var[e_idx] == val)
            problem.solve(solver=cp.GUROBI, warm_start=False)  
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        # Extract final y_line
        y_opt = y_var.value
        out_path = os.path.join(self.results_folder, f"{gid}_{mode}_y.npy")
        np.save(out_path, y_opt)
        return gid, y_opt

    def run(self, mode: str = "warmstart", num_workers: int = None):
        if num_workers is None:
            num_workers = 2 # max(1, os.cpu_count() - 1)
        print(f"Running optimization in '{mode}' mode with {num_workers} workers")

        # pool = mp.Pool(processes=num_workers)
        args_iterable = [(i, mode) for i in range(len(self.graph_ids))]
        # # Use tqdm to wrap the pool.map for progress tracking
        # results = pool.map(self._optimize_single, args_iterable)
        # pool.close()
        # pool.join()

        # with mp.Pool(processes=num_workers) as pool:
        #     # Use imap_unordered (or imap) so that we can iterate over results one at a time
        #     it = pool.imap_unordered(self._optimize_single, args_iterable)

        #     results = {}
        #     for gid, y_opt in tqdm(it,
        #                             total=len(self.graph_ids),
        #                             desc="Optimizing networks",
        #                             unit="network",
        #                             leave=False):
        #         results[gid] = y_opt

        # apply serialized processing
        results = {}
        for idx in tqdm(range(len(self.graph_ids)), desc="Optimizing networks", unit="network", leave=False):
            gid, y_opt = self._optimize_single((idx, mode))
            results[gid] = y_opt

        return dict(results)


if __name__ == "__main__":
    import argparse
    import numpy as np
    import cvxpy as cp

    parser = argparse.ArgumentParser(description="Predict-then-Optimize Pipeline")
    parser.add_argument("--config_path",type=str,default=None,help="Path to the YAML config file. If not provided, will attempt to auto-detect.",)
    parser.add_argument("--model_path", type=str,default = r"model_search\models\MLP\clear-monkey-40-Best.pt", help="Path to pretrained GNN checkpoint")
    parser.add_argument("--folder_names",type=str,default = [r"data\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32\test"],help="Folder containing 'mst' and 'mst_opt' subfolders",)
    parser.add_argument("--dataset_names",type=str,default = ["test"],help="Three folders for train/validation/test (each must have 'mst', 'mst_opt')",    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for Predictor (use 1 to simplify warmstart indexing)")
    parser.add_argument("--results_folder",type=str,default="predict_opt_results",help="Where to store optimization outputs",)
    parser.add_argument("--mode",type=str,choices=["warmstart", "threshold"],default="warmstart",help="Whether to run warmstart-based solve or threshold-fixed solve")
    parser.add_argument("--threshold",type=float,default=0.5,help="Threshold for rounding in 'threshold' mode",)
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

        # 3) Run Optimizer
        optimizer = Optimizer(
            folder_name =args.folder_names[0],  # Use the first folder name for the network data
            warmstart_path=(full_warmstart_path),
            results_folder=(results_base),
            threshold=args.threshold,
        )
        final_results = optimizer.run(mode=args.mode, num_workers=args.num_workers)

        summary_path = results_base / "summary_results.pkl"
        with open(summary_path, "wb") as f:
            pickle.dump(final_results, f)
        print(f"Summary saved to: {summary_path}")
