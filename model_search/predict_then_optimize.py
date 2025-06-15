import sys
import os
import torch
import torch.multiprocessing as mp
from torch_geometric.data import DataLoader
import pandapower as pp
import yaml
import importlib
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import pandas as pd
import json
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyomo.environ import value as pyo_val
from src.SOCP_class_dnr import SOCP_class
from model_search.evaluation.evaluation import load_config_from_model_path

from model_search.models.AdvancedMLP.AdvancedMLP import PhysicsInformedRounding
from data_generation.define_ground_truth import is_radial_and_connected

def load_pp_networks(base_directory):
    nets = {"mst": {}, "mst_opt": {}}
    for phase in ["mst", "mst_opt"]:
        folder = os.path.join(base_directory, phase, "pandapower_networks")
        if not os.path.isdir(folder):
            continue
        for fn in tqdm(os.listdir(folder), desc=f"Loading {phase} networks from {folder}"):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(folder, fn)
            try:
                net = pp.from_json(path)
            except:
                with open(path) as f:
                    raw = f.read()
                if raw.startswith('"') and raw.endswith('"'):
                    raw = json.loads(raw)
                try:
                    net = pp.from_json_string(raw)
                except:
                    net = from_json_dict(json.loads(raw))
            if net.bus.empty:
                continue
            nets[phase][fn] = net
    return nets
def collapse_switches_to_one_per_line(net):
    """
    Collapse multiple switches per line to one representative switch per line.
    Returns the collapsed switch dataframe and any conflicting lines.
    """
    # Get line switches only
    line_switches = net.switch[net.switch['et'] == 'l'].copy()
    
    # Detect any conflicting states
    conflicts = (
        line_switches.groupby("element")["closed"]
        .nunique()
        .loc[lambda x: x > 1]
        .index
        .tolist()
    )
    
    if conflicts:
        print(f"Warning: Lines with conflicting switch states: {conflicts}")
    
    # Collapse to one switch per line (keep first)
    collapsed_switches = line_switches.sort_index().drop_duplicates(subset="element", keep="first")
    
    return collapsed_switches, conflicts

def extract_edge_index_from_network(net):
  
    # Collapse switches to one per line
    collapsed_switches, _ = collapse_switches_to_one_per_line(net)

    
    edge_list = []
    valid_line_indices = []
    
    for idx, (_, switch_row) in enumerate(collapsed_switches.iterrows()):
        line_idx = switch_row['element']
        
        # Check if the line exists
        if line_idx in net.line.index:
            line = net.line.loc[line_idx]
            from_bus = line['from_bus']
            to_bus = line['to_bus']
            
            # Add ONE edge per line (not bidirectional)
            edge_list.append([from_bus, to_bus])
            valid_line_indices.append(line_idx)
            
            print(f"Debug: Line {line_idx}: {from_bus} -> {to_bus}")
        else:
            print(f"Warning: Switch references non-existent line {line_idx}")
    
    if not edge_list:
        print("Error: No valid edges found in network")
        return torch.tensor([[0], [0]], dtype=torch.long), []
    
    # Convert to tensor format [2, E]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    print(f"Debug: Created edge_index with shape {edge_index.shape}")
    print(f"Debug: edge_list length: {len(edge_list)}")
    print(f"Debug: Expected: {len(collapsed_switches)} edges = {len(collapsed_switches)} unique lines")
    
    assert edge_index.size(1) == len(collapsed_switches), f"Edge count mismatch: {edge_index.size(1)} edges vs {len(collapsed_switches)} unique lines"
    
    return edge_index, valid_line_indices

def apply_physics_informed_rounding(switch_probs, net, device='cpu'):
    """
    Apply physics-informed rounding using actual network topology.
    Each switch probability should correspond to exactly one line.
    """
    print(f"Debug: Input switch_probs length: {len(switch_probs)}")
    
    # Convert to tensor if needed
    if not isinstance(switch_probs, torch.Tensor):
        switch_probs = torch.tensor(switch_probs, dtype=torch.float32, device=device)
    
    # Extract actual network topology
    edge_index, valid_line_indices = extract_edge_index_from_network(net)
    edge_index = edge_index.to(device)
    
    print(f"Debug: Extracted {edge_index.size(1)} edges")
    print(f"Debug: edge_index dtype: {edge_index.dtype}, device: {edge_index.device}")
    
    # Verify we have the right number of switches
    collapsed_switches, _ = collapse_switches_to_one_per_line(net)
    expected_switches = len(collapsed_switches)
    
    if len(switch_probs) != expected_switches:
        raise ValueError(f"Switch count mismatch: GNN predicted {len(switch_probs)} switches, "
                        f"but network has {expected_switches} unique lines with switches")
    
    if edge_index.size(1) != len(switch_probs):
        raise ValueError(f"Edge count mismatch: extracted {edge_index.size(1)} edges, "
                        f"but have {len(switch_probs)} switch probabilities")
    
    # Apply PhysicsInformedRounding
    try:
        # Use the fixed PhysicsInformedRounding class
        phyr = PhysicsInformedRounding()
        
        # Single graph case - ensure all tensors have correct dtype
        edge_batch = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
        num_nodes_per_graph = torch.tensor([edge_index.max().item() + 1], dtype=torch.long, device=device)
        
        print(f"Debug: Applying PhyR with {len(switch_probs)} switches and {edge_index.size(1)} edges")
        print(f"Debug: switch_probs dtype: {switch_probs.dtype}, shape: {switch_probs.shape}")
        print(f"Debug: edge_batch dtype: {edge_batch.dtype}, shape: {edge_batch.shape}")
        print(f"Debug: num_nodes_per_graph dtype: {num_nodes_per_graph.dtype}, value: {num_nodes_per_graph}")
        
        # Ensure switch_probs is 1D
        if switch_probs.dim() > 1:
            switch_probs = switch_probs.squeeze()
        
        # Apply physics rounding
        binary_decisions = phyr(switch_probs, edge_index, edge_batch, num_nodes_per_graph)
        
        # Handle output format
        if isinstance(binary_decisions, torch.Tensor):
            binary_decisions = binary_decisions.cpu().numpy()
        
        print(f"Debug: PhyR returned {len(binary_decisions)} binary decisions")
        
        # Convert to int list
        result = [int(decision) for decision in binary_decisions]
        
        print(f"Applied PhysicsInformedRounding to {len(result)} switches")
        return result
        
    except Exception as e:
        print(f"PhysicsInformedRounding failed: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to simple rounding")
        return [1 if prob > 0.5 else 0 for prob in switch_probs.cpu().numpy()]

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

        print(f"Config keys: {list(config.keys())}")
        print(f"Model kwargs from config: {config.get('model_kwargs', {})}")
        print(f"Output type: {config.get('output_type', 'not found')}")
        print(f"Num classes: {config.get('num_classes', 'not found')}")

        self.eval_args = argparse.Namespace(**config)
        
        sample_data = sample_loader.dataset[0]
        node_input_dim = sample_data.x.shape[1]
        edge_input_dim = sample_data.edge_attr.shape[1]
        
        print(f"Data dimensions inferred:")
        print(f"  Node input dim: {node_input_dim}")
        print(f"  Edge input dim: {edge_input_dim}\n")

        model_module = importlib.import_module(f"models.{self.eval_args.model_module}.{self.eval_args.model_module}")
        model_class = getattr(model_module, self.eval_args.model_module)
        
        # Load the checkpoint first to check its structure
        print(f"Loading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Check the output dimension of the switch head in the checkpoint
        switch_head_key = "switch_head.1.weight"  
        if switch_head_key in state_dict:
            saved_output_dim = state_dict[switch_head_key].shape[0]
            print(f"Saved model switch head output dimension: {saved_output_dim}")
            
            # Override config based on saved model structure
            if saved_output_dim == 1:
                print("Detected binary output model - overriding config to binary mode")
                config["output_type"] = "binary"
                config["num_classes"] = 2
            elif saved_output_dim == 2:
                print("Detected multiclass output model - using multiclass mode")
                config["output_type"] = "multiclass"
                config["num_classes"] = 2
            else:
                print(f"Unexpected output dimension: {saved_output_dim}, using config defaults")
        
        # Create model with corrected configuration
        base_kwargs = config.get("model_kwargs", {})
        self.model = model_class(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_type=config.get("output_type", "binary"),
            num_classes=config.get("num_classes", 2),
            **base_kwargs
        ).to(self.device)
        
        print(f"Model architecture:")
        print(self.model)

        # Check if the state dict keys match
        model_keys = set(self.model.state_dict().keys())
        saved_keys = set(state_dict.keys())
        missing_keys = model_keys - saved_keys
        unexpected_keys = saved_keys - model_keys

        if missing_keys:
            print(f"Missing keys in saved model: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in saved model: {unexpected_keys}")
            
        print(f"State dict loaded successfully: {len(missing_keys) == 0 and len(unexpected_keys) == 0}")
        
        # Load state dict
        self.model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully!")
        self.model.eval()

    def run(self, dataloader: DataLoader, graph_ids=None):
        predictions = {}
        
        if graph_ids is None:
            if hasattr(dataloader.dataset, 'graph_ids'):
                graph_ids = dataloader.dataset.graph_ids
            else:
                graph_ids = list(range(len(dataloader.dataset)))
        
        sample_idx = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting switch scores", unit="batch", leave=False):
                batch = batch.to(self.device)
                output = self.model(batch)
                

                if "switch_probabilities" in output:
                    probabilities = output["switch_probabilities"][:, 1]
                    print(f"Using multiclass predictions (class 1 probabilities)")
                elif "switch_predictions" in output:
                    probabilities = output["switch_predictions"]
                    print(f"Using binary predictions")
                else:
                    logits = output.get("switch_logits")
                    if logits is None:
                        raise RuntimeError("Model output must contain 'switch_logits', 'switch_predictions', or 'switch_probabilities'")
                    if logits.shape[-1] == 1:
                        probabilities = torch.sigmoid(logits).squeeze(-1)
                        print(f"Applied sigmoid to single output logits (shape: {logits.shape})")
                    elif logits.shape[-1] == 2:
                        probabilities = torch.softmax(logits, dim=-1)[:, 1]
                        print(f"Applied softmax to one-hot logits (shape: {logits.shape}), taking class 1")
                    else:
                        raise RuntimeError(f"Unexpected logits shape: {logits.shape}. Expected last dimension to be 1 or 2.")
                
                probabilities = probabilities.detach().cpu().numpy()
                
                print(f"Final probabilities shape: {probabilities.shape}")
                print(f"Batch size from dataloader: {batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1}")
                
                if hasattr(batch, 'batch'):  
                    batch_size = batch.batch.max().item() + 1

        
                    edge_to_graph = batch.batch[batch.edge_index[0]] 
    
                    for graph_idx in range(batch_size):
                        if sample_idx < len(graph_ids):
                            graph_edge_mask = (edge_to_graph == graph_idx).cpu().numpy()
                            graph_probs = probabilities[graph_edge_mask]
                            
                            graph_id = graph_ids[sample_idx]
                            predictions[graph_id] = graph_probs.tolist()
                            
                            print(f"Graph {graph_id}: {len(graph_probs)} switch probabilities (range: {min(graph_probs):.3f}-{max(graph_probs):.3f})")
                            sample_idx += 1
                else:
                    # Single graph case
                    if sample_idx < len(graph_ids):
                        graph_id = graph_ids[sample_idx]
                        predictions[graph_id] = probabilities.tolist()
                        
                        print(f"Graph {graph_id}: {len(probabilities)} switch probabilities (range: {min(probabilities):.3f}-{max(probabilities):.3f})")
                        sample_idx += 1
        
        print(f"Predicted switch probabilities for {len(predictions)} samples")
        return predictions

class WarmstartSOCP(SOCP_class):
    """Enhanced SOCP class with warmstart support"""
    
    def __init__(self, net, graph_id: str = "", *,
                 switch_penalty: float = 0.0001,
                 slack_penalty: float = 0.1,
                 voltage_slack_penalty: float = 0.1,
                 load_shed_penalty: float = 100,
                 toggles=None,
                 debug_level=0,
                 active_bus_mask=None,
                 fixed_switches=None, 
                 float_warmstart=None):
        self.fixed_switches = fixed_switches or {}
        self.float_warmstart = float_warmstart

        super().__init__(
            net=net,
            graph_id=graph_id,
            switch_penalty=switch_penalty,
            slack_penalty=slack_penalty,
            voltage_slack_penalty=voltage_slack_penalty,
            load_shed_penalty=load_shed_penalty,
            toggles=toggles,
            debug_level=debug_level,
            active_bus_mask=active_bus_mask
        )
    
    def initialize(self):
        """Override to handle fixed switches"""
        super().initialize()

        if self.fixed_switches:
            print(f"Fixing {len(self.fixed_switches)} switches based on warmstart")
            for switch_idx, desired_status in self.fixed_switches.items():
                if switch_idx < len(self.switch_df):
                    self.switch_df.loc[switch_idx, 'closed'] = bool(desired_status)
                    print(f"Fixed switch {switch_idx} to {'ON' if desired_status else 'OFF'}")
    
    def create_model(self):
        """Override to apply warmstart and fix switches"""
        model = super().create_model()
        

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
            switch_idx = 0
            for l in m.lines:
                if l in self.lines_with_sw and switch_idx < len(self.float_warmstart):
                    float_value = max(0.0, min(1.0, float(self.float_warmstart[switch_idx])))
                    m.line_status[l].set_value(float_value)
                    print(f"Float warmstart: line {l} = {float_value:.3f}")
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
                        print(f"Float warmstart: switch {s} = {float_value:.3f}")
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
            "TimeLimit": 6000, 
            "MIPGap": 1e-2,
            "NonConvex": 2,
        })
        opt.options.update(solver_kw)
        
        self.solver_results = opt.solve(tee=self.debug_level > 1, load_solutions=True)
        self.solve_time = time.time() - start
        
        obj_val = pyo_val(self.model.objective)
        print("Solved with float warmstart in %.2fs (obj = %.3f)", self.solve_time, obj_val)
        
        return self.solver_results


class ExplicitSaver:
    """Enhanced ExplicitSaver with better rounding support"""
    
    def __init__(self, root_folder: str, model_name: str, warmstart_mode: str, rounding_method: str):
        self.root_folder = Path(root_folder)
        self.model_name = model_name
        self.warmstart_mode = warmstart_mode
        self.rounding_method = rounding_method
        
        # Create predictions folder in root
        self.predictions_folder = self.root_folder / "predictions"
        self.predictions_folder.mkdir(exist_ok=True)
        
        # Create specific folders for warmstart and prediction networks
        self.warmstart_folder = self.predictions_folder / f"warm-start-{model_name}-{rounding_method}-{warmstart_mode}"
        self.prediction_folder = self.predictions_folder / f"prediction-{model_name}-{rounding_method}-{warmstart_mode}"
        
        # Create pandapower_networks subfolders
        self.warmstart_networks_folder = self.warmstart_folder / "pandapower_networks"
        self.prediction_networks_folder = self.prediction_folder / "pandapower_networks"
        
        # Create all folders
        for folder in [self.warmstart_folder, self.prediction_folder, 
                      self.warmstart_networks_folder, self.prediction_networks_folder]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV data storage
        self.csv_data = []
        
        # Set device for physics rounding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    
    def extract_ground_truth(self, pp_networks: dict, graph_id: str) -> list:
        """Extract binary ground truth from unique lines only"""
        mst_opt_net = pp_networks["mst_opt"][graph_id]
        mst_net = pp_networks["mst"][graph_id]
        
        # Collapse switches to one per line
        mst_collapsed, _ = collapse_switches_to_one_per_line(mst_net)
        opt_collapsed, _ = collapse_switches_to_one_per_line(mst_opt_net)
        
        # Ensure same order by line element
        mst_lines = sorted(mst_collapsed['element'].unique())
        opt_lines = sorted(opt_collapsed['element'].unique())
        
        if mst_lines != opt_lines:
            print(f"Warning: Line mismatch between mst and opt networks for graph {graph_id}")
        
        # Extract states for each unique line
        ground_truth = []
        for line_idx in mst_lines:
            if line_idx in opt_collapsed['element'].values:
                state = opt_collapsed[opt_collapsed['element'] == line_idx]['closed'].iloc[0]
                ground_truth.append(int(state))
            else:
                print(f"Warning: Line {line_idx} not found in opt network")
                ground_truth.append(0)  # Default to open
        
        return ground_truth
    
    def extract_initial_state(self, pp_networks: dict, graph_id: str) -> list:
        """Extract initial state from mst network - unique lines only"""
        mst_net = pp_networks["mst"][graph_id]
        collapsed_switches, _ = collapse_switches_to_one_per_line(mst_net)
        
        # Sort by line element to ensure consistent order
        collapsed_switches = collapsed_switches.sort_values('element')
        return [int(closed) for closed in collapsed_switches['closed'].values]
    
    def apply_rounding(self, predictions: list, method: str = "round", network=None) -> list:
        """Apply rounding method to predictions"""
        if method == "round":
            return [1 if pred > 0.5 else 0 for pred in predictions]
        elif method == "PhyR":
            return apply_physics_informed_rounding(predictions, network, self.device)
        else:
            raise ValueError(f"Unknown rounding method: {method}")
    
    def save_network_as_json(self, net: pp.pandapowerNet, folder: Path, graph_id: str):
        """Save pandapower network as JSON"""
        filepath = folder / f"{graph_id}.json"
        pp.to_json(net, str(filepath))
    
    def add_csv_entry(self, graph_id: str,ground_truth: list,initial_state: list,gnn_probs: list,gnn_prediction: list,warmstart_config: list,final_optima: list,
                     solve_time: float,objective: float,radial: bool,connected: bool,pf_converged: bool,
                     switches_changed: int,gt_loss: float,pred_loss: float):              # predicted network loss
        
        # Find the minimum length to ensure consistency
        lengths = [len(arr) for arr in [ground_truth, initial_state, gnn_prediction, warmstart_config, final_optima]]
        self.csv_data.append({
            'graph_id':           graph_id,
             'ground_truth':       json.dumps(ground_truth),
            'initial_state':      json.dumps(initial_state),
            'gnn_probs':          json.dumps(gnn_probs),
            'gnn_prediction':     json.dumps(gnn_prediction),
            'warmstart_config':   json.dumps(warmstart_config),
            'final_optima':       json.dumps(final_optima),
            'solve_time':         solve_time,
            'objective':          objective,
            'radial':             radial,
            'connected':          connected,
            'pf_converged':       pf_converged,
            'switches_changed':   switches_changed,
            'gt_loss':            gt_loss,
            'pred_loss':          pred_loss,
            # flag for “same loss but different switch pattern”
            'same_loss_diff_topo': abs(gt_loss - pred_loss) < 1e-6
        })

    def save_csv(self):
        """Save CSV file with all data"""
        csv_path = self.predictions_folder / f"results-{self.model_name}-{self.rounding_method}-{self.warmstart_mode}.csv"
        df = pd.DataFrame(self.csv_data)
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")
        return csv_path


class Optimizer:
    def __init__(self, 
                 folder_name: str,
                 predictions: dict,
                 saver: ExplicitSaver,
                 warmstart_mode: str = "none",
                 confidence_threshold: float = 0.8):
        
        self.folder_name = folder_name
        self.predictions = predictions
        self.saver = saver
        self.warmstart_mode = warmstart_mode
        self.confidence_threshold = confidence_threshold
        self.toggles = {
            "include_voltage_drop_constraint": True,
            "include_voltage_bounds_constraint": True,
            "include_power_balance_constraint": True,
            "include_radiality_constraints": True,
            "use_spanning_tree_radiality": False,   
            "use_root_flow": True,
            "include_switch_penalty": True,
            "include_cone_constraint": True,
            "all_lines_are_switches": True,
            "allow_load_shed": True,
        }

        # Load networks
        self.pp_all = load_pp_networks(self.folder_name)

        self.graph_ids = sorted(self.pp_all["mst"].keys())
        
        # Verify predictions match networks
        prediction_ids = set(self.predictions.keys())
        network_ids = set(self.graph_ids)
        
        if prediction_ids != network_ids:
            missing_in_predictions = network_ids - prediction_ids
            extra_in_predictions = prediction_ids - network_ids
            
            if missing_in_predictions:
                print(f"Warning: Networks without predictions: {missing_in_predictions}")
            if extra_in_predictions:
                print(f"Warning: Predictions without networks: {extra_in_predictions}")
            
            # Only process networks that have both data
            self.graph_ids = sorted(list(network_ids & prediction_ids))
            print(f"Processing {len(self.graph_ids)} networks with matching prediction data")

    def _optimize_single(self, args):
        idx, mode = args
        gid = self.graph_ids[idx]
        net = self.pp_all["mst"][gid].deepcopy()
        raw_preds = self.predictions[gid]

        # Apply rounding for soft/float so that net is initialized correctly
        rounded_preds = self.saver.apply_rounding(raw_preds, self.saver.rounding_method, network=net)

        # Prepare fixed_switches if needed
        fixed_switches = {}

        try:
            # === 1) WarmStart Setup ===
            if mode == "soft":
                # apply rounded_preds directly to net.switch
                from data_generation.define_ground_truth import collapse_switches_to_one_per_line
                collapsed, _ = collapse_switches_to_one_per_line(net)
                for i, (_, row) in enumerate(collapsed.iterrows()):
                    if i < len(rounded_preds):
                        mask = (net.switch.et=="l") & (net.switch.element==row.element)
                        net.switch.loc[mask, "closed"] = bool(rounded_preds[i])
                self.saver.save_network_as_json(net, self.saver.warmstart_networks_folder, gid)
                optimizer = WarmstartSOCP(net=net, toggles=self.toggles, graph_id=gid)

            elif mode == "float":
                optimizer = WarmstartSOCP(
                    net=net,
                    toggles=self.toggles,
                    graph_id=gid,
                    float_warmstart=raw_preds
                )
                # also save a copy with discrete rounding
                warm_net = net.deepcopy()
                from data_generation.define_ground_truth import collapse_switches_to_one_per_line
                collapsed, _ = collapse_switches_to_one_per_line(warm_net)
                for i, (_, row) in enumerate(collapsed.iterrows()):
                    if i < len(rounded_preds):
                        mask = (warm_net.switch.et=="l") & (warm_net.switch.element==row.element)
                        warm_net.switch.loc[mask, "closed"] = bool(rounded_preds[i])
                self.saver.save_network_as_json(warm_net, self.saver.warmstart_networks_folder, gid)

            elif mode == "hard":
                # compute thresholds
                T = self.confidence_threshold
                half = T/2
                upper = 0.5 + half
                lower = 0.5 - half

                from data_generation.define_ground_truth import collapse_switches_to_one_per_line
                collapsed, _ = collapse_switches_to_one_per_line(net)
                total = len(collapsed)
                fixed = 0

                # gather stats
                probs = raw_preds[:total]
                pos = [p for p in probs if p>=upper]
                neg = [p for p in probs if p<=lower]
                avg_p1 = float(np.mean(pos)) if pos else 0.0
                avg_p0 = float(np.mean(neg)) if neg else 0.0

                for i, (_, row) in enumerate(collapsed.iterrows()):
                    if i>=len(raw_preds): break
                    s = raw_preds[i]
                    # scalar-sigmoid case
                    p = float(s)
                    if p>=upper:
                        d=1
                    elif p<=lower:
                        d=0
                    else:
                        continue
                    # fix it
                    mask = (net.switch.et=="l") & (net.switch.element==row.element)
                    net.switch.loc[mask,"closed"] = bool(d)
                    fixed += 1
                    fixed_switches.update({idx:d for idx in net.switch[mask].index})

                self.saver.save_network_as_json(net, self.saver.warmstart_networks_folder, gid)
                optimizer = WarmstartSOCP(
                    net=net,
                    toggles=self.toggles,
                    graph_id=gid,
                    fixed_switches=fixed_switches
                )

                # summary print
                pct = fixed/total*100 if total else 0.0
                print(f"[hard] Fixed {fixed}/{total} lines ({pct:.1f}%) | avg(p1)={avg_p1:.3f} | avg(p0)={avg_p0:.3f}")

            else:  # mode == "none"
                self.saver.save_network_as_json(net, self.saver.warmstart_networks_folder, gid)
                optimizer = SOCP_class(net=net, toggles=self.toggles, graph_id=gid)

            # === 2) Solve ===
            optimizer.initialize()
            res = optimizer.solve(solver="gurobi_persistent", TimeLimit=300, MIPGap=1e-3)

            # === 3) Extract metrics ===
            # final switch states (one per line)
            from data_generation.define_ground_truth import collapse_switches_to_one_per_line
            final_net = self.pp_all["mst"][gid].deepcopy()
            # apply solution back into final_net
            sol = {l: round(pyo_val(optimizer.model.line_status[l])) for l in optimizer.model.lines}
            collapsed, _ = collapse_switches_to_one_per_line(final_net)
            final_states = []
            for row in collapsed.itertuples():
                val = sol.get(row.element, 0)
                mask = (final_net.switch.et=="l") & (final_net.switch.element==row.element)
                final_net.switch.loc[mask,"closed"] = bool(val)
                final_states.append(int(val))

            # power flow
            try:
                pp.runpp(final_net, enforce_q_lims=False)
                pred_loss = float(final_net.res_line["pl_mw"].sum())
                pf_ok = final_net.converged
            except:
                pred_loss, pf_ok = float("nan"), False

            # ground-truth optimum loss
            gt_net = self.pp_all["mst_opt"][gid]
            try:
                pp.runpp(gt_net, enforce_q_lims=False)
                gt_loss = float(gt_net.res_line["pl_mw"].sum())
            except:
                gt_loss = float("nan")

            # radial/connectivity
            radial, connected = is_radial_and_connected(final_net, include_switches=True)
            flips = sum(1 for g,p in zip(self.saver.extract_ground_truth(self.pp_all,gid), final_states) if g!=p)

            # === 4) CSV entry ===
            self.saver.add_csv_entry(
                graph_id=gid,
                ground_truth=self.saver.extract_ground_truth(self.pp_all, gid),
                initial_state=self.saver.extract_initial_state(self.pp_all, gid),
                gnn_probs=raw_preds,
                gnn_prediction=rounded_preds,
                warmstart_config=rounded_preds,
                final_optima=final_states,
                solve_time=optimizer.solve_time,
                objective=pyo_val(optimizer.model.objective),
                radial=radial,
                connected=connected,
                pf_converged=pf_ok,
                switches_changed=flips,
                gt_loss=gt_loss,
                pred_loss=pred_loss
            )

            # === 5) return ===
            result = {
                'solution': sol,
                'objective': pyo_val(optimizer.model.objective),
                'solve_time': optimizer.solve_time,
                'switches_changed': flips,
                'warmstart_mode': mode,
                'success': True
            }
            print(f"[{mode}] Finished graph {gid} in {optimizer.solve_time:.2f}s, success=True")
            return gid, result

        except Exception as e:
            print(f"Error optimizing network {gid}: {e}")
            import traceback; traceback.print_exc()
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

        # Save CSV after all optimizations
        csv_path = self.saver.save_csv()
        
        return results, csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict-then-Optimize Pipeline with Explicit Saving")
    parser.add_argument("--config_path", type=str,
                        default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\model_search\models\AdvancedMLP\config_files\AdvancedMLP------jumping-wave-13.yaml",
                       help="Path to the YAML config file")
    parser.add_argument("--model_path", type=str,
                       default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\model_search\models\AdvancedMLP\jumping-wave-13-Best.pt", 
                       help="Path to pretrained GNN checkpoint")
    parser.add_argument("--folder_names", type=str, nargs="+",
                       default=[r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\split_datasets\test_test"],
                       help="Folder containing 'mst' and 'mst_opt' subfolders")
    parser.add_argument("--dataset_names", type=str, nargs="+",
                       default=["test"],
                       help="Dataset names corresponding to folder_names")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Batch size for Predictor")
    
    # Warmstart arguments
    parser.add_argument("--warmstart_mode", type=str,
                       choices=["none", "soft", "float", "hard"],
                       default="hard",
                       help="Warmstart strategy")
    parser.add_argument("--rounding_method", type=str,
                       choices=["round", "PhyR"],
                       default="round",
                       help="Rounding method for predictions")
    parser.add_argument("--confidence_threshold", type=float, default=1,
                       help="Confidence threshold for hard warmstart")
    
    parser.add_argument("--num_workers", type=int, default=0.01,
                       help="Number of CPU workers for optimization")
    parser.add_argument("--predict", action="store_true", default=True, 
                       help="Run prediction step")
    parser.add_argument("--optimize", action="store_true", default=True, 
                       help="Run optimization step")
    
    args = parser.parse_args()

    # Build DataLoaders
    from load_data import create_data_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pp_networks = load_pp_networks(args.folder_names[0])
    print(pp_networks)
    graph_ids = sorted(pp_networks["mst"].keys())

    loaders = create_data_loaders(
        dataset_names=args.dataset_names,
        folder_names=args.folder_names,
        dataset_type="cvx",
        batch_size=args.batch_size,
        batching_type="standard",
    )
    test_loader = loaders.get("test", None)
    
    # Verify alignment
    if len(test_loader.dataset) != len(graph_ids):
        print(f"Warning: Dataset size ({len(test_loader.dataset)}) != number of graphs ({len(graph_ids)})")
    
    print(f"Test loader created with {len(test_loader.dataset)} samples")
    print(f"Graph IDs: {graph_ids[:5]}..." if len(graph_ids) > 5 else f"Graph IDs: {graph_ids}")

    # Extract job name from model_path
    job_name = Path(args.model_path).stem
    print(f"Job name extracted from model path: {job_name}")

    # Initialize explicit saver
    saver = ExplicitSaver(
        root_folder=args.folder_names[0],
        model_name=job_name,
        warmstart_mode=args.warmstart_mode,
        rounding_method=args.rounding_method
    )

    predictions = None
    if args.predict:            
        print("Starting prediction...")
        predictor = Predictor(
            model_path=args.model_path,
            config_path=args.config_path,
            device=device,
            sample_loader=test_loader
        )
        predictions = predictor.run(test_loader, graph_ids=graph_ids)

    if args.optimize and predictions is not None:
        print(f"Starting optimization with '{args.warmstart_mode}' warmstart...")

        optimizer = Optimizer(
            folder_name=args.folder_names[0],  
            predictions=predictions,
            saver=saver,
            warmstart_mode=args.warmstart_mode,
            confidence_threshold=args.confidence_threshold,
        )
        final_results, csv_path = optimizer.run(num_workers=args.num_workers)

        print(f"Results saved to CSV: {csv_path}")
        print(f"Warmstart networks saved to: {saver.warmstart_networks_folder}")
        print(f"Final optimized networks saved to: {saver.prediction_networks_folder}")
        
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
        
        print(f"\nFolder structure created:")
        print(f"  Root predictions folder: {saver.predictions_folder}")
        print(f"  Warmstart folder: {saver.warmstart_folder}")
        print(f"  Final predictions folder: {saver.prediction_folder}")
        print(f"  CSV file: {csv_path}")
    
    elif args.optimize and predictions is None:
        print("Cannot run optimization without predictions. Run prediction step first.")