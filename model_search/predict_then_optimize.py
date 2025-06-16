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
import matplotlib.pyplot as plt
import logging
import pandas as pd
import json
import numpy as np
import time 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyomo.environ import value as pyo_val
from src.SOCP_class_dnr import SOCP_class
from model_search.evaluation.evaluation import load_config_from_model_path

from model_search.models.AdvancedMLP.AdvancedMLP import PhysicsInformedRounding
from data_generation.define_ground_truth import is_radial_and_connected
from model_search.load_data2 import *

class NetworkSwitchManager:
    def __init__(self, net):
        self.net = net
        self.collapsed_switches, self.conflicts = collapse_switches_to_one_per_line(net)
        self.line_order = sorted(
            int(x) for x in self.collapsed_switches["element"].tolist()
        )
        self.line_to_switches_map = self._build_line_to_switches_map()

        self.state_history = []
        self.state_labels = []

        # Capture initial state immediately
        self.initial_state = self._capture_current_state()
        self.add_state(self.initial_state, "initial")

    def _build_line_to_switches_map(self):
        """Build mapping from line_id to all switch indices that control that line"""
        line_switches = self.net.switch[self.net.switch['et'] == 'l'].copy()
        mapping = {}
        for switch_idx, switch_row in line_switches.iterrows():
            line_id = switch_row['element']
            if line_id not in mapping:
                mapping[line_id] = []
            mapping[line_id].append(switch_idx)
        return mapping
    
    def _capture_current_state(self):
        """Capture the current state directly from the network"""
        states = []
        for line_id in self.line_order:
            idxs = self.line_to_switches_map.get(line_id, [])
            if idxs:
                states.append(int(self.net.switch.at[idxs[0], 'closed']))
            else:
                states.append(0)
        return states

    def get_initial_states(self):
        """Get the initial states that were captured at creation time"""
        return self.initial_state.copy()
    
    def get_switch_states(self):
        """Get current switch states as list (one per line) - reads from actual network state"""
        return self._capture_current_state()
    
    def set_switch_states(self, states, label=None):
        """Set switch states for all lines and optionally track with label"""
        if len(states) != len(self.collapsed_switches):
            raise ValueError(f"Expected {len(self.collapsed_switches)} states, got {len(states)}")
            
        # Then update the actual network
        sorted_switches = self.collapsed_switches.sort_values("element")
        for i, (_, switch_row) in enumerate(sorted_switches.iterrows()):
            line_id = switch_row['element']
            new_state = bool(states[i])
            
            # Update ALL switches that control this line
            switch_indices = self.line_to_switches_map.get(line_id, [])
            for switch_idx in switch_indices:
                self.net.switch.at[switch_idx, 'closed'] = new_state

        # First 
        if label:
            self.add_state(states, label)
    
    def add_state(self, states, label):
        """Add a state to the history for visualization"""
        # Check for duplicate labels and skip if already exists
        if label in self.state_labels:
            print(f"Warning: State '{label}' already exists, skipping duplicate")
            return
            
        self.state_history.append(list(states))
        self.state_labels.append(label)
        
        # Debug output only for important states
        print(f"Added state '{label}': {sum(states)} closed, {len(states) - sum(states)} open switches")
    
    def get_ground_truth_states(self, opt_net):
        """Extract ground truth states from optimized network"""
        opt_manager = NetworkSwitchManager(opt_net)
        opt_collapsed = opt_manager.collapsed_switches
        
        gt_states = []
        for line_id in self.line_order:
            if line_id in opt_collapsed["element"].values:
                gt_state = int(opt_collapsed.loc[opt_collapsed["element"] == line_id, "closed"].iloc[0])
            else:
                gt_state = 0
            gt_states.append(gt_state)
        return gt_states
    
    def _build_graph_for_plotting(self):
        G = nx.Graph()
        G.add_nodes_from(self.net.bus.index)

        optimized = set(self.line_order)              # <-- use same list
        for idx, ln in self.net.line.iterrows():
            lid = int(idx)
            if lid in optimized:
                G.add_edge(int(ln.from_bus), int(ln.to_bus),
                        line_id=lid, is_switch=True)
            else:
                G.add_edge(int(ln.from_bus), int(ln.to_bus),
                        line_id=lid, is_switch=False)
        return G
    
    def _get_positions(self, G):
        """Get node positions for plotting"""
        if not self.net.bus_geodata.empty:
            pos = {int(b): (r.x, r.y) for b, r in self.net.bus_geodata.iterrows()}
        else:
            pos = nx.spring_layout(G, seed=0)  
        return pos
    
    def plot_state_comparison(self, figsize=(15, 4), save_path=None):
        """Plot comparison of all tracked states with gradient coloring"""
        if not self.state_history:
            print("No states to plot")
            return
        
        n_states = len(self.state_history)
        fig, axes = plt.subplots(1, n_states, figsize=figsize)
        
        if n_states == 1:
            axes = [axes]
        
        G = self._build_graph_for_plotting()
        pos = self._get_positions(G)
        
        # Create colormap for gradient (red to green: closed to open)
        import matplotlib.cm as cm
        cmap = cm.RdYlGn_r  # Red-Yellow-Green reversed (red=closed, green=open)
        
        for i, (states, label) in enumerate(zip(self.state_history, self.state_labels)):
            ax = axes[i]
            
            # Check if this is probability data (values between 0 and 1 but not exactly 0 or 1)
            is_probability = label == "gnn_probs" or any(0 < s < 1 for s in states if isinstance(s, (int, float)))
            
            regular_edges = []
            switch_edges = []
            switch_values = []

            # Build a mapping from line_id -> state using our canonical order
            line_states = {lid: states[i] for i, lid in enumerate(self.line_order)}

            for u, v, data in G.edges(data=True):
                line_id = data['line_id']
                if data['is_switch']:
                    switch_edges.append((u, v))
                    switch_values.append(line_states.get(line_id, 0))
                else:
                    regular_edges.append((u, v))

            # Debug print
            if switch_values:
                avg_val = np.mean(switch_values)
                n_closed = sum(1 for v in switch_values if v > 0.5)
                n_open = len(switch_values) - n_closed
                print(f"Plotting '{label}': {len(switch_values)} switches, avg={avg_val:.3f}, closed={n_closed}, open={n_open}")
            # Draw the graph - edges first, then nodes
            # Regular edges in grey (draw first, under everything)
            if regular_edges:
                nx.draw_networkx_edges(G, pos, edgelist=regular_edges, 
                                    edge_color="lightgrey", width=1.0, ax=ax)
            
            # Draw switches with gradient colors (draw second, over regular edges)
            if switch_edges and switch_values:
                switch_values = np.array(switch_values)
                
                # Draw each switch edge individually with its color
                for edge, value in zip(switch_edges, switch_values):
                    color = cmap(value)  # Get color from colormap
                    
                    # Check if this is hard warmstart and if this switch is fixed
                    is_fixed = hasattr(self, 'fixed_switches') and label == "hard_warmstart"
                    
                    if is_fixed:
                        # Draw double line for fixed switches
                        nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                            edge_color=[color], width=4.0, ax=ax, alpha=0.8)
                        nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                            edge_color=['white'], width=2.0, ax=ax, alpha=0.8)
                        nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                            edge_color=[color], width=1.0, ax=ax, alpha=0.8)
                    else:
                        # Regular switch edge
                        nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                            edge_color=[color], width=2.5, ax=ax, alpha=0.8)
            
            # All nodes in grey (draw last, on top of edges)
            nx.draw_networkx_nodes(G, pos, node_size=20, node_color="lightgrey", 
                                edgecolors="grey", linewidths=0.3, ax=ax)
            
            # Count switches
            if is_probability:
                # For probabilities, show average value
                avg_value = np.mean(states)
                ax.set_title(f"{label}\navg prob: {avg_value:.2f}")
            else:
                # For binary states, count open/closed
                n_closed = sum(states)
                n_open = len(states) - n_closed
                ax.set_title(f"{label}\n{n_closed:.1f} closed, {n_open:.1f} open")
            
            ax.axis("off")
        
        # Add colorbar to show the gradient scale
        if switch_edges:  # Only add colorbar if there are switches
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', 
                            shrink=0.6, pad=0.1, aspect=30)
            cbar.set_label('Switch State (0=Open, 1=Closed)', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()
    
    def plot_single_state(self, states=None, label="Current State", ax=None, show_labels=False, fixed_switches=None):
        """Plot a single state of the network with gradient coloring"""
        if states is None:
            states = self.get_switch_states()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            show_plot = True
        else:
            show_plot = False
        
        G = self._build_graph_for_plotting()
        pos = self._get_positions(G)
        
        # Create colormap for gradient (red to green: closed to open)
        import matplotlib.cm as cm
        cmap = cm.RdYlGn_r  # Red-Yellow-Green reversed (red=closed, green=open)
        
        # Separate regular edges and switch edges with their values
        regular_edges = []
        switch_edges = []
        switch_values = []
        
        line_states = {}
        sorted_switches = self.collapsed_switches.sort_values("element")
        for i, (_, switch_row) in enumerate(sorted_switches.iterrows()):
            line_id = switch_row['element']
            line_states[line_id] = states[i] if i < len(states) else 0
        
        for u, v, data in G.edges(data=True):
            line_id = data['line_id']
            if data['is_switch']:
                switch_edges.append((u, v))
                switch_values.append(line_states.get(line_id, 0))
            else:
                regular_edges.append((u, v))
        
        # Draw the graph - edges first, then nodes
        # Regular edges in grey (draw first, under everything)
        if regular_edges:
            nx.draw_networkx_edges(G, pos, edgelist=regular_edges, 
                                 edge_color="lightgrey", width=1.0, ax=ax)
        
        # Draw switches with gradient colors (draw second, over regular edges)
        if switch_edges and switch_values:
            switch_values = np.array(switch_values)
            
            # Draw each switch edge individually with its color
            for edge, value in zip(switch_edges, switch_values):
                color = cmap(value)  # Get color from colormap
                
                # Check if this switch is fixed (for hard warmstart)
                is_fixed = fixed_switches is not None and any(
                    line_id in self.line_to_switches_map and 
                    any(sw_idx in fixed_switches for sw_idx in self.line_to_switches_map[line_id])
                    for line_id in [data['line_id'] for u, v, data in G.edges(data=True) if (u, v) == edge]
                )
                
                if is_fixed:
                    # Draw double line for fixed switches
                    nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                         edge_color=[color], width=4.0, ax=ax, alpha=0.8)
                    nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                         edge_color=['white'], width=2.0, ax=ax, alpha=0.8)
                    nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                         edge_color=[color], width=1.0, ax=ax, alpha=0.8)
                else:
                    # Regular switch edge
                    nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                         edge_color=[color], width=2.5, ax=ax, alpha=0.8)
        
        # All nodes in grey (draw last, on top of edges)
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color="lightgrey", 
                             edgecolors="grey", linewidths=0.3, ax=ax)
        
        # Optionally show node labels
        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        # Count switches
        n_closed = sum(states)
        n_open = len(states) - n_closed
        
        ax.set_title(f"{label}\n{n_closed:.1f} closed, {n_open:.1f} open")
        ax.axis("off")
        
        if show_plot:
            plt.tight_layout()
            plt.show()


def collapse_switches_to_one_per_line(net):
    line_switches = net.switch[net.switch['et'] == 'l'].copy()
    conflicts = (
        line_switches.groupby("element")["closed"]
        .nunique()
        .loc[lambda x: x > 1]
        .index
        .tolist()
    )
    if conflicts:
        print(f"Warning: Lines with conflicting switch states: {conflicts}")
    collapsed_switches = (
        line_switches.sort_index()
        .drop_duplicates(subset="element", keep="first")
    )
    return collapsed_switches, conflicts


def apply_physics_informed_rounding(switch_probs, switch_manager, device='cpu'):
    """Updated to use NetworkSwitchManager"""
    if not isinstance(switch_probs, torch.Tensor):
        switch_probs = torch.tensor(switch_probs, dtype=torch.float32, device=device)

    # Build edge index from collapsed switches
    edge_list = []
    for _, switch_row in switch_manager.collapsed_switches.iterrows():
        line_idx = switch_row['element']
        if line_idx in switch_manager.net.line.index:
            line = switch_manager.net.line.loc[line_idx]
            edge_list.append([line.from_bus, line.to_bus])

    if not edge_list:
        return [0] * len(switch_probs)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
    expected = len(switch_manager.collapsed_switches)
    
    if len(switch_probs) != expected:
        raise ValueError(f"Switch count mismatch: GNN predicted {len(switch_probs)}, network has {expected}")

    phyr = PhysicsInformedRounding()
    edge_batch = torch.zeros(expected, dtype=torch.long, device=device)
    num_nodes = torch.tensor([edge_index.max().item() + 1], dtype=torch.long, device=device)
    decisions = phyr(switch_probs.squeeze(), edge_index, edge_batch, num_nodes)

    if isinstance(decisions, torch.Tensor):
        decisions = decisions.cpu().numpy()
    return [int(x) for x in decisions]

class Predictor:
    def __init__(self, model_path, config_path, device, sample_loader):
        self.device = device
        config = (yaml.safe_load(open(config_path)) if config_path
                  else load_config_from_model_path(model_path))
        self.eval_args = argparse.Namespace(**config)
        self.gnn_times = {}
        sample = sample_loader.dataset[0]
        node_dim = sample.x.shape[1]
        edge_dim = sample.edge_attr.shape[1]

        module = importlib.import_module(
            f"models.{self.eval_args.model_module}.{self.eval_args.model_module}"
        )
        cls = getattr(module, self.eval_args.model_module)

        ckpt = torch.load(model_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)

        head = state.get("switch_head.1.weight")
        if head is not None:
            dim = head.shape[0]
            if dim == 1:
                print("Detected binary classification head, setting output type accordingly.")
                config["output_type"] = "binary"
                config["num_classes"] = 2
            elif dim == 2:
                print("Detected multiclass classification head, setting output type accordingly.")
                config["output_type"] = "multiclass"
                config["num_classes"] = 2

        self.model = cls(
            node_input_dim=node_dim,
            edge_input_dim=edge_dim,
            output_type=config.get("output_type", "binary"),
            num_classes=config.get("num_classes", 2),
            **config.get("model_kwargs", {})
        ).to(device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.loader = sample_loader

    def run(self, loader=None, graph_ids=None):
            loader = loader or self.loader
            preds = {}
            batch_counter = 0  # Track which graph we're processing
            print(f"graph_ids: {graph_ids}")
            print(f"Predicting on {len(loader.dataset)} samples with {self.eval_args.model_module} model...")
            with torch.no_grad():
                for batch in tqdm(loader, desc="Predicting", leave=False):
                    start_time = time.time()
                    batch = batch.to(self.device)
                    out = self.model(batch)
                    if "switch_probabilities" in out:
                        probs = out["switch_probabilities"][..., 1]
                    elif "switch_predictions" in out:
                        probs = out["switch_predictions"]
                    else:
                        logits = out.get("switch_logits")
                        if logits.shape[-1] == 1:
                            probs = torch.sigmoid(logits).squeeze(-1)
                        else:
                            probs = torch.softmax(logits, dim=-1)[..., 1]
                    probs = probs.cpu().numpy()
                    gnn_time = time.time() - start_time
                    preds[graph_ids[batch_counter]] = probs.tolist() 
                    self.gnn_times[graph_ids[batch_counter]] = gnn_time
                    batch_counter += 1
                    

            print(f"Predictions complete. Total graphs: {len(preds)}")
            print(f"Prediction keys: {list(preds.keys())}")
            if preds:
                first_key = list(preds.keys())[0]
                print(f"Example predictions for first graph ({first_key}): {preds[first_key][:5]}...")
            
            return preds, self.gnn_times

class WarmstartSOCP(SOCP_class):
    def __init__(self, net, graph_id="", **kwargs):
        super().__init__(net=net, graph_id=graph_id, **kwargs)
        self.fixed_switches = kwargs.get('fixed_switches', {})
        self.float_warmstart = kwargs.get('float_warmstart')

    def initialize(self):
        super().initialize()
        for idx, val in self.fixed_switches.items():
            self.switch_df.loc[idx, 'closed'] = bool(val)
    
    def create_model(self):
        model = super().create_model()
        for idx, val in self.fixed_switches.items():
            row = self.switch_df.iloc[idx]
            if row.et == 'l' and row.element in model.lines:
                model.line_status[row.element].fix(val)
        return model
    
    def solve(self, solver="gurobi_persistent", **opts):
        if self.float_warmstart is not None:
            return self._solve_with_float_warmstart(solver, **opts)
        return super().solve(solver=solver, **opts)

    def _solve_with_float_warmstart(self, solver, **opts):
        from pyomo.opt import SolverFactory
        if self.model is None:
            self.create_model()
        m = self.model
        if hasattr(m, 'model_switches'):
            for i, s in enumerate(m.model_switches):
                if i < len(self.float_warmstart):
                    m.switch_status[s].set_value(
                        float(self.float_warmstart[i]))
        opt = SolverFactory(solver)
        if hasattr(opt, 'set_instance'):
            opt.set_instance(m)
        opt.options.update(opts)
        res = opt.solve(tee=False, load_solutions=True)
        self.solve_time = 0.0
        return res


class ExplicitSaver:
    """Updated ExplicitSaver with fixed state tracking"""
    
    def __init__(self, root_folder: str, model_name: str, warmstart_mode: str, 
                 rounding_method: str, confidence_threshold: float = None):
        self.root_folder = Path(root_folder)
        self.model_name = model_name
        self.warmstart_mode = warmstart_mode
        self.rounding_method = rounding_method
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup folders (same as before)
        suffix = f"{warmstart_mode}-{confidence_threshold}" if warmstart_mode == "hard" and confidence_threshold is not None else warmstart_mode
        self.predictions_folder = self.root_folder / "predictions"
        self.predictions_folder.mkdir(exist_ok=True)
        self.warmstart_folder = self.predictions_folder / f"warm-start-{model_name}-{rounding_method}-{suffix}"
        self.prediction_folder = self.predictions_folder / f"prediction-{model_name}-{rounding_method}-{suffix}"
        self.warmstart_networks_folder = self.warmstart_folder / "pandapower_networks"
        self.prediction_networks_folder = self.prediction_folder / "pandapower_networks"
        
        # Add visualization folder
        self.visualization_folder = self.predictions_folder / f"visualizations-{model_name}-{rounding_method}-{suffix}"
        
        for folder in (self.warmstart_folder, self.prediction_folder, 
                      self.warmstart_networks_folder, self.prediction_networks_folder,
                      self.visualization_folder):
            folder.mkdir(parents=True, exist_ok=True)
        
        self.csv_data = []
        self.graph_managers = {}

    def extract_ground_truth(self, pp_networks: dict, graph_id: str, switch_manager: NetworkSwitchManager) -> list:
        """Extract ground truth using NetworkSwitchManager - NO STATE TRACKING HERE"""
        opt_net = pp_networks["mst_opt"][graph_id]
        gt_states = switch_manager.get_ground_truth_states(opt_net)
        return gt_states

    def extract_initial_state(self, switch_manager: NetworkSwitchManager) -> list:
        """Extract initial state using NetworkSwitchManager"""
        return switch_manager.get_switch_states()


    def apply_rounding(self, predictions: list, method: str, switch_manager: NetworkSwitchManager) -> list:
        """Apply rounding using NetworkSwitchManager - NO STATE TRACKING HERE"""
        if method == "round":
            rounded = [1 if p > 0.5 else 0 for p in predictions]
        elif method == "PhyR":
            rounded = apply_physics_informed_rounding(predictions, switch_manager, device=self.device)
        else:
            raise ValueError(f"Unknown rounding method: {method}")
        return rounded

    def save_network_as_json(self,
                             net: pp.pandapowerNet,
                             folder: Path,
                             graph_id: str):
        path = folder / f"{graph_id}.json"
        pp.to_json(net, str(path))

    def save_graph_visualization(self, graph_id: str, switch_manager: NetworkSwitchManager):
        """Save visualization for a specific graph"""
        vis_path = self.visualization_folder / f"{graph_id}_state_evolution.png"
        switch_manager.plot_state_comparison(save_path=vis_path)
        print(f"Saved state evolution visualization: {vis_path}")

    def save_all_visualizations(self):
        """Save visualizations for all collected graphs"""
        for graph_id, manager in self.graph_managers.items():
            self.save_graph_visualization(graph_id, manager)


    def add_csv_entry(self, graph_id: str, ground_truth: list, initial_state: list,
                  gnn_probs: list, gnn_prediction: list, warmstart_config: list,
                  final_optima: list, solve_time: float, objective: float,
                  radial: bool, connected: bool, pf_converged: bool,
                  switches_changed: int, gt_loss: float, pred_loss: float,
                  gnn_time: float = 0.0,  # New parameter
                  error_message: str = None):
        """Add CSV entry with separate timing columns"""
        
        # Calculate total time based on warmstart mode
        if self.warmstart_mode == "optimization_without_warmstart":
            total_time = solve_time 
        elif self.warmstart_mode == "only_gnn_predictions":
            total_time = gnn_time  
        else:
            total_time = gnn_time + solve_time  
        
        self.csv_data.append({
            "graph_id": graph_id,
            "ground_truth": json.dumps(ground_truth),
            "initial_state": json.dumps(initial_state),
            "gnn_probs": json.dumps(gnn_probs),
            "gnn_prediction": json.dumps(gnn_prediction),
            "warmstart_config": json.dumps(warmstart_config),
            "final_optima": json.dumps(final_optima),
            "gnn_time": gnn_time,  # New column
            "solve_time": solve_time,  # Existing column
            "total_time": total_time,  # New column
            "objective": objective,
            "radial": radial,
            "connected": connected,
            "pf_converged": pf_converged,
            "switches_changed": switches_changed,
            "gt_loss": gt_loss,
            "pred_loss": pred_loss,
            "same_loss_diff_topo": abs(gt_loss - pred_loss) < 1e-6 if (gt_loss is not None and pred_loss is not None) else False,
            "error": error_message or ""
        })


    def save_csv(self) -> Path:
        suffix = f"{self.warmstart_mode}-{self.confidence_threshold}" if self.warmstart_mode == "hard" and self.confidence_threshold is not None else self.warmstart_mode
        csv_path = self.predictions_folder / f"results-{self.model_name}-{self.rounding_method}-{suffix}.csv"
        pd.DataFrame(self.csv_data).to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")
        return csv_path



class Optimizer:
    def __init__(self, folder_name: str, predictions: dict, saver: ExplicitSaver,
                 warmstart_mode: str = "optimization_without_warmstart",
                 confidence_threshold: float = 0.8,
                 gnn_times: dict = None):
        self.folder_name = folder_name
        self.predictions = predictions
        self.saver = saver
        self.warmstart_mode = warmstart_mode
        self.confidence_threshold = confidence_threshold
        self.gnn_times = gnn_times or {}

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

        self.pp_all = load_pp_networks(self.folder_name)
        self.graph_ids = sorted(self.pp_all["mst"].keys())

        # align predictions/network IDs
        preds = set(self.predictions.keys())
        nets  = set(self.graph_ids)
        if preds != nets:
            common = sorted(preds & nets)
            print(f"Processing only {len(common)} matching graphs")
            self.graph_ids = common


    def _optimize_single(self, args):
        idx, mode = args
        gid = self.graph_ids[idx]
        gnn_time = self.gnn_times.get(gid, 0.0)
        try:
            # Create network and manager
            net = self.pp_all["mst"][gid].deepcopy()
            switch_manager = NetworkSwitchManager(net)
            
            # Store manager for visualization
            self.saver.graph_managers[gid] = switch_manager
            
            # 1. Initial state is already captured in manager constructor
            initial_state = switch_manager.get_initial_states()
            
            # 3. GNN probabilities
            raw_preds = self.predictions[gid]
            switch_manager.add_state(raw_preds, "gnn_probs")
            
            # 4. GNN rounded predictions
            rounded = self.saver.apply_rounding(raw_preds, self.saver.rounding_method, switch_manager)
            switch_manager.add_state(rounded, f"gnn_round")
            
            # Apply rounded predictions to network
            switch_manager.set_switch_states(rounded)

            # --- Warmstart setup ---
            if mode == "soft":
                # Network already has rounded states applied
                radial, connected = is_radial_and_connected(net, include_switches=True)
                print(f"Soft warmstart {gid}: {rounded.count(0)} zeros/{rounded.count(1)} ones, radial={radial}, connected={connected}")
                self.saver.save_network_as_json(net, self.saver.warmstart_networks_folder, gid)
                solver = WarmstartSOCP(net=net, toggles=self.toggles, graph_id=gid)

            elif mode == "float":
                solver = WarmstartSOCP(net=net, toggles=self.toggles, graph_id=gid, float_warmstart=raw_preds)
                # also record discrete version
                net2 = net.deepcopy()
                switch_manager2 = NetworkSwitchManager(net2)
                switch_manager2.set_switch_states(rounded, "float_warmstart_discrete")
                self.saver.save_network_as_json(net2, self.saver.warmstart_networks_folder, gid)
                print(f"Float warmstart {gid}: using raw predictions as float warmstart ")
                print(f"Amount of switches rounded to 0/1: {rounded.count(0)}/{rounded.count(1)}")
                print(f"Average value of switches rounded to 0 : {np.mean([p for p in raw_preds if p < 0.5]):.2f} , 1: {np.mean([p for p in raw_preds if p >= 0.5]):.2f}")
            
            elif mode == "hard":
                T     = self.confidence_threshold
                half  = T / 2
                up, lo = 0.5 + half, 0.5 - half
                fixed = {}
                fixed_0_count = 0
                fixed_1_count = 0
                sorted_switches = switch_manager.collapsed_switches.sort_values("element")
                for i, (_, switch_row) in enumerate(sorted_switches.iterrows()):
                    if i >= len(raw_preds): break
                    p = raw_preds[i]
                    line_id = switch_row['element']
                    if p >= up:
                        d = 1
                        fixed_1_count += 1
                    elif p <= lo:
                        d = 0
                        fixed_0_count += 1
                    else:
                        continue
                    switch_indices = switch_manager.line_to_switches_map.get(line_id, [])
                    for switch_idx in switch_indices:
                        fixed[switch_idx] = d
                        net.switch.at[switch_idx, 'closed'] = bool(d)

                switch_manager.set_switch_states(rounded, "hard_warmstart")

                radial, connected = is_radial_and_connected(net, include_switches=True)
                print(f"Hard warmstart {gid}: {fixed_0_count} fixed to 0, {fixed_1_count} fixed to 1, radial={radial}, connected={connected}")
                
                self.saver.save_network_as_json(net, self.saver.warmstart_networks_folder, gid)
                solver = WarmstartSOCP(net=net, toggles=self.toggles, graph_id=gid, fixed_switches=fixed)


            elif mode == "only_gnn_predictions":
                # No optimization, just use GNN predictions
                radial, connected = is_radial_and_connected(net, include_switches=True)
                print(f"GNN-only {gid}: {rounded.count(0)} zeros/{rounded.count(1)} ones, radial={radial}, connected={connected}")
                self.saver.save_network_as_json(net, self.saver.warmstart_networks_folder, gid)
                solver = None
            else:  
                switch_manager.set_switch_states(initial_state)
                self.saver.save_network_as_json(net, self.saver.warmstart_networks_folder, gid)
                solver = SOCP_class(net=net, toggles=self.toggles, graph_id=gid)

            if solver is not None:
                solver.initialize()
                solver.solve(solver="gurobi_persistent", TimeLimit=300, MIPGap=1e-3)
                solve_time = solver.solve_time
                objective_value = float(pyo_val(solver.model.objective))
                
                # Extract solution and apply to network
                final_net = self.pp_all["mst"][gid].deepcopy()
                final_switch_manager = NetworkSwitchManager(final_net)
                
                sol = {l: round(pyo_val(solver.model.line_status[l])) for l in solver.model.lines}
                final_states = []
                for _, switch_row in final_switch_manager.collapsed_switches.sort_values("element").iterrows():
                    v = sol.get(switch_row['element'], 0)
                    final_states.append(int(v))
                
                # Apply solution to ALL switches in the network
                final_switch_manager.set_switch_states(final_states)

                switch_manager.set_switch_states(final_states, "optimization_result")
            else:
                # GNN-only mode: no optimization
                solve_time = 0.0
                objective_value = float("nan")
                final_net = net  # Already modified with GNN predictions
                final_states = rounded
                switch_manager.add_state(final_states, "gnn_only_final")

            # power‐flow losses
            try:
                pp.runpp(final_net, enforce_q_lims=False)
                pred_loss = float(final_net.res_line.pl_mw.sum())
                pf_ok = final_net.converged
            except:
                pred_loss, pf_ok = float("nan"), False

            # ground‐truth loss
            gt_net = self.pp_all["mst_opt"][gid]
            try:
                pp.runpp(gt_net, enforce_q_lims=False)
                gt_loss = float(gt_net.res_line.pl_mw.sum())
            except:
                gt_loss = float("nan")

            ground_truth = switch_manager.get_ground_truth_states(self.pp_all["mst_opt"][gid])
            switch_manager.add_state(ground_truth, "ground_truth")

            radial, connected = is_radial_and_connected(final_net, include_switches=True)
            flips = sum(1 for g, f in zip(ground_truth, final_states) if g != f)
            
            print(f"\nGraph {gid} states collected: {switch_manager.state_labels}")
            print(f"Final: radial={radial}, connected={connected}, switches changed={flips}")

    
            self.saver.add_csv_entry(
                graph_id=gid,
                ground_truth=ground_truth,
                initial_state=initial_state,  # Use the captured initial state
                gnn_probs=raw_preds,
                gnn_prediction=rounded,
                warmstart_config=rounded,
                final_optima=final_states,
                gnn_time=gnn_time,  # Add this
                solve_time=solve_time,
                objective=objective_value,
                radial=radial,
                connected=connected,
                pf_converged=pf_ok,
                switches_changed=flips,
                gt_loss=gt_loss,
                pred_loss=pred_loss,
                error_message=None
            )
            
            

            print("current switch states:", switch_manager.get_switch_states())
            print("labels: ", switch_manager.state_labels)

            self.saver.save_network_as_json(final_net, self.saver.prediction_networks_folder, gid)
        
            return gid, {'success': True}

        except Exception as e:
            print(f"ERROR in graph {gid}: {str(e)}") 
            self.saver.add_csv_entry(
                graph_id=gid,
                ground_truth=[],
                initial_state=[],
                gnn_probs=self.predictions.get(gid, []),
                gnn_prediction=[],
                warmstart_config=[],
                final_optima=[],
                gnn_time=0.0,
                solve_time=0.0,
                objective=float("nan"),
                radial=False,
                connected=False,
                pf_converged=False,
                switches_changed=0,
                gt_loss=float("nan"),
                pred_loss=float("nan"),
                error_message=str(e)
            )
            return gid, {'success': False, 'error': str(e)}

    def run(self, num_workers: int = None):
        if num_workers is None:
            num_workers = max(1, os.cpu_count() - 1)
        print(f"Running optimization ({self.warmstart_mode}) with {num_workers} workers")

        args = [(i, self.warmstart_mode) for i in range(len(self.graph_ids))]
        results = {}

        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                for gid, res in pool.imap_unordered(self._optimize_single, args):
                    results[gid] = res
        else:
            for arg in args:
                gid, res = self._optimize_single(arg)
                results[gid] = res

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
    
    # Warmstart arguments
    parser.add_argument("--warmstart_mode", type=str,
                        choices=["only_gnn_predictions", "soft", "float", "hard", "optimization_without_warmstart"],
                        default="only_gnn_predictions",
                        help="Warmstart strategy")
    parser.add_argument("--rounding_method", type=str,
                        choices=["round", "PhyR"],
                        default="round",
                        help="Rounding method for predictions")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Confidence threshold for hard warmstart")
    
    parser.add_argument("--num_workers", type=int, default=0,
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
        dataset_type="default",
        batch_size=1)
    test_loader = loaders.get("test", None)
    
    # Verify alignment
    if len(test_loader.dataset) != len(graph_ids):
        print(f"Warning: Dataset size ({len(test_loader.dataset)}) != number of graphs ({len(graph_ids)})")
    
    print(f"Test loader created with {len(test_loader.dataset)} samples")
    print(f"Graph IDs: {graph_ids[:5]}..." if len(graph_ids) > 5 else f"Graph IDs: {graph_ids}")

    # Extract job name from model_path
    job_name = Path(args.model_path).stem
    print(f"Job name extracted from model path: {job_name}")

    # Initialize explicit saver with confidence threshold
    saver = ExplicitSaver(root_folder=args.folder_names[0],
        model_name=job_name,
        warmstart_mode=args.warmstart_mode,
        rounding_method=args.rounding_method,
        confidence_threshold=args.confidence_threshold if args.warmstart_mode == "hard" else None)

    predictions = None
    gnn_times = None
    if args.predict:         
        print(f"\n ===================================================\n       RUN PREDICTIONS \n =================================================== \n ")   
        print("Starting prediction...")
        predictor = Predictor(
            model_path=args.model_path,
            config_path=args.config_path,
            device=device,
            sample_loader=test_loader
        )
        predictions, gnn_times = predictor.run(test_loader, graph_ids=graph_ids)

        print(f"Prediction keys: {sorted(predictions.keys())}")
        print(f"Graph IDs: {sorted(graph_ids)}")
        print(f"Predictions type: {type(list(predictions.keys())[0]) if predictions else 'None'}")
        print(f"Graph IDs type: {type(graph_ids[0]) if graph_ids else 'None'}")

        
    if args.optimize and predictions is not None:
        print(f"\n ===================================================\n        RUN OPTIMIZATION \n =================================================== \n ")
        print(f"Starting optimization with '{args.warmstart_mode}' warmstart...")

        optimizer = Optimizer(
            folder_name=args.folder_names[0],  
            predictions=predictions,
            saver=saver,
            warmstart_mode=args.warmstart_mode,
            confidence_threshold=args.confidence_threshold,
            gnn_times=gnn_times,
        )
        final_results, csv_path = optimizer.run(num_workers=args.num_workers)

        # Save visualizations for all graphs
        print("\nSaving visualizations for all graphs...")
        saver.save_all_visualizations()
        print(f"Visualizations saved to: {saver.visualization_folder}")

        print(f"Optimization completed with {len(final_results)} results.")
        print(f"\n ===================================================\n       PROCESS RESULTS\n =================================================== \n ")
        if csv_path:
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
        if total > successful:
            print(f"  Failed solves: {total - successful}")
        print(f"  Average solve time: {avg_solve_time:.2f}s")
        print(f"  Average switches changed: {avg_switches_changed:.1f}")
        if args.warmstart_mode == "hard":
            print(f"  Confidence threshold: {args.confidence_threshold}")
        
        print(f"\nFolder structure created:")
        print(f"  Root predictions folder: {saver.predictions_folder}")
        print(f"  Warmstart folder: {saver.warmstart_folder}")
        print(f"  Final predictions folder: {saver.prediction_folder}")
        if csv_path:
            print(f"  CSV file: {csv_path}")
    
    elif args.optimize and predictions is None:
        print("Cannot run optimization without predictions. Run prediction step first.")