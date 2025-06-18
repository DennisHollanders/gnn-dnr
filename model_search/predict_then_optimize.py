from random import shuffle
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
        self.collapsed_switches, self.conflicts = self._create_all_line_switches(net)
        self.line_order = sorted(int(lid) for lid in self.net.line.index)
        self.line_to_switches_map = self._build_line_to_switches_map()

        self.state_history = []
        self.state_labels = []

        self.initial_state = self._capture_current_state()
        self.add_state(self.initial_state, "initial")

    def _create_all_line_switches(self, net):
        """Create a switch representation for ALL lines, matching data loader behavior"""
        # Get actual switches
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
        
        actual_switches = (
            line_switches.sort_index()
            .drop_duplicates(subset="element", keep="first")
        )
        
        all_line_switches = []
        for line_id in sorted(net.line.index):
            if line_id in actual_switches["element"].values:
                switch_row = actual_switches[actual_switches["element"] == line_id].iloc[0]
                all_line_switches.append({
                    "element": line_id,
                    "closed": switch_row["closed"],
                    "has_physical_switch": True
                })
            else:
                all_line_switches.append({
                    "element": line_id,
                    "closed": True, 
                    "has_physical_switch": False
                })
        
   
        all_switches_df = pd.DataFrame(all_line_switches)
        return all_switches_df, conflicts

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
                # real switch: read its closed status
                states.append(int(self.net.switch.at[idxs[0], 'closed']))
            else:
                # no switch on this line → treat as “closed” by default (1)
                states.append(1)
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
            
        # Update the actual network
        sorted_switches = self.collapsed_switches.sort_values("element")
        for i, (_, switch_row) in enumerate(sorted_switches.iterrows()):
            line_id = switch_row['element']
            new_state = bool(states[i])
            
            # Only update if this line has physical switches
            if switch_row.get('has_physical_switch', True):
                # Update ALL switches that control this line
                switch_indices = self.line_to_switches_map.get(line_id, [])
                for switch_idx in switch_indices:
                    self.net.switch.at[switch_idx, 'closed'] = new_state
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
                gt_state = 1  # Default to closed for lines without switches
            gt_states.append(gt_state)
        return gt_states
    
    def _build_graph_for_plotting(self):
        G = nx.Graph()
        G.add_nodes_from(self.net.bus.index)

        optimized = set(self.line_order)
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
    def plot_state_comparison(self, figsize=(15, 4), save_path=None, fixed_switches=None):
        """Plot comparison of all tracked states with gradient coloring
        
        Args:
            figsize: Figure size
            save_path: Path to save figure
            fixed_switches: Dict of {line_id: value} for fixed switches (optional)
        """
        if not self.state_history:
            print("No states to plot")
            return
        
        n_states = len(self.state_history)
        fig, axes = plt.subplots(1, n_states, figsize=figsize)
        
        if n_states == 1:
            axes = [axes]
        
        G = self._build_graph_for_plotting()
        pos = self._get_positions(G)
        
        # Use a colormap with strong contrast between 0 and 1
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Create custom colormap: red for open (0), green for closed (1)
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "green"])
        
        for i, (states, label) in enumerate(zip(self.state_history, self.state_labels)):
            ax = axes[i]
            
            # Create line_id -> state mapping
            line_to_state_idx = {lid: idx for idx, lid in enumerate(self.line_order)}
            line_states = {lid: states[line_to_state_idx[lid]] for lid in self.line_order}
            
            # Track nodes connected to open switches
            open_switch_nodes = set()
            
            # For gnn_probs, use gradient coloring
            if label == "gnn_probs":
                # Draw non-switch edges first
                for u, v, data in G.edges(data=True):
                    if not data['is_switch']:
                        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                            edge_color="lightgrey", width=1.0, ax=ax, alpha=0.3)
                
                # Draw switch edges with gradient colors
                for u, v, data in G.edges(data=True):
                    line_id = data['line_id']
                    if data['is_switch']:
                        state_value = line_states.get(line_id, 0)
                        color = cmap(state_value)  # This will scale 0-1 automatically
                        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                            edge_color=[color], width=3.0, ax=ax, alpha=0.9)
                        
                        if state_value < 0.5:
                            open_switch_nodes.add(u)
                            open_switch_nodes.add(v)
            else:
                # For binary states, separate edges by type
                switch_edges_open = []
                switch_edges_closed = []
                switch_edges_fixed_open = []
                switch_edges_fixed_closed = []
                non_switch_edges = []
                
                # Categorize edges
                for u, v, data in G.edges(data=True):
                    line_id = data['line_id']
                    
                    if data['is_switch']:
                        state_value = line_states.get(line_id, 0)
                        
                        # Check if this switch is fixed
                        is_fixed = fixed_switches and line_id in fixed_switches
                        
                        if state_value < 0.5:
                            if is_fixed:
                                switch_edges_fixed_open.append((u, v))
                            else:
                                switch_edges_open.append((u, v))
                            open_switch_nodes.add(u)
                            open_switch_nodes.add(v)
                        else:
                            if is_fixed:
                                switch_edges_fixed_closed.append((u, v))
                            else:
                                switch_edges_closed.append((u, v))
                    else:
                        non_switch_edges.append((u, v))
                
                # Draw edges with appropriate colors
                # Non-switch edges (thin grey)
                if non_switch_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=non_switch_edges,
                                        edge_color="lightgrey", width=1.0, ax=ax, alpha=0.3)
                
                # Open switches (thick red, solid line)
                if switch_edges_open:
                    nx.draw_networkx_edges(G, pos, edgelist=switch_edges_open,
                                        edge_color="red", width=3.0, ax=ax, alpha=0.9)
                
                # Closed switches (thick green)
                if switch_edges_closed:
                    nx.draw_networkx_edges(G, pos, edgelist=switch_edges_closed,
                                        edge_color="green", width=3.0, ax=ax, alpha=0.9)
                
                # Fixed open switches (thick red, double dashed)
                if switch_edges_fixed_open:
                    nx.draw_networkx_edges(G, pos, edgelist=switch_edges_fixed_open,
                                        edge_color="red", width=3.0, ax=ax, alpha=0.9,
                                        style=(0, (5, 2, 1, 2)))  # double dash pattern
                    nx.draw_networkx_edges(G, pos, edgelist=switch_edges_fixed_open,
                                        edge_color="black", width=1.0, ax=ax, alpha=0.9,
                                        style=(0, (5, 2, 1, 2)))  # double dash pattern
                
                # Fixed closed switches (thick green, double dashed)
                if switch_edges_fixed_closed:
                    nx.draw_networkx_edges(G, pos, edgelist=switch_edges_fixed_closed,
                                        edge_color="green", width=3.0, ax=ax, alpha=0.9,
                                        style=(0, (5, 2, 1, 2)))  # double dash pattern
                    nx.draw_networkx_edges(G, pos, edgelist=switch_edges_fixed_closed,
                                        edge_color="black", width=1.0, ax=ax,
                                        style=(0, (5, 2, 1, 2)))  # double dash pattern
            
            # Draw nodes
            node_colors = []
            for node in G.nodes():
                if node in open_switch_nodes:
                    node_colors.append("red")
                else:
                    node_colors.append("lightgrey")
            
            nx.draw_networkx_nodes(G, pos, node_size=30, node_color=node_colors,
                                edgecolors="black", linewidths=0.5, ax=ax)
            
            # Set title with counts
            if label == "gnn_probs":
                avg_value = np.mean(states)
                title = f"{label}\navg prob: {avg_value:.2f}"
            else:
                n_closed = sum(1 for s in states if s > 0.5)
                n_open = len(states) - n_closed
                title = f"{label}\n{n_closed} closed, {n_open} open"
            
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        
        # Add colorbar for probability visualization
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', 
                        shrink=0.6, pad=0.1, aspect=30)
        cbar.set_label('Switch State (0=Open/Red, 1=Closed/Green)', fontsize=10)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=3, label='Closed Switch'),
            Line2D([0], [0], color='red', lw=3, label='Open Switch'),
            Line2D([0], [0], color='red', lw=3, linestyle=(0, (5, 2, 1, 2)), label='Fixed Open Switch'),
            Line2D([0], [0], color='green', lw=3, linestyle=(0, (5, 2, 1, 2)), label='Fixed Closed Switch'),
            Line2D([0], [0], color='lightgrey', lw=1, label='Non-switch Line'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.15))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()
    # def plot_state_comparison(self, figsize=(15, 4), save_path=None):
    #     """Plot comparison of all tracked states with gradient coloring"""
    #     if not self.state_history:
    #         print("No states to plot")
    #         return
        
    #     n_states = len(self.state_history)
    #     fig, axes = plt.subplots(1, n_states, figsize=figsize)
        
    #     if n_states == 1:
    #         axes = [axes]
        
    #     G = self._build_graph_for_plotting()
    #     pos = self._get_positions(G)
        
    #     # Create colormap for gradient (red=closed, green=open)
    #     import matplotlib.cm as cm
    #     cmap = cm.RdYlGn_r  
        
    #     for i, (states, label) in enumerate(zip(self.state_history, self.state_labels)):
    #         ax = axes[i]
            
    #         # Create line_id -> state mapping
    #         line_states = {lid: states[j] for j, lid in enumerate(self.line_order)}
            
    #         # Separate edges by type and get their colors
    #         for u, v, data in G.edges(data=True):
    #             line_id = data['line_id']
                
    #             if data['is_switch']:
    #                 # Get the state value for this switch
    #                 state_value = line_states.get(line_id, 0)
    #                 color = cmap(state_value)
                    
    #                 # Draw switch edge with color based on state
    #                 nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
    #                                     edge_color=[color], width=2.5, ax=ax, alpha=0.8)
    #             else:
    #                 # Draw non-switch edge in grey
    #                 nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
    #                                     edge_color="lightgrey", width=1.0, ax=ax)
            
    #         # Draw all nodes
    #         nx.draw_networkx_nodes(G, pos, node_size=20, node_color="lightgrey", 
    #                             edgecolors="grey", linewidths=0.3, ax=ax)
            
    #         # Set title based on state type
    #         if label == "gnn_probs":
    #             avg_value = np.mean(states)
    #             ax.set_title(f"{label}\navg prob: {avg_value:.2f}")
    #         else:
    #             n_closed = sum(1 for s in states if s > 0.5)
    #             n_open = len(states) - n_closed
    #             ax.set_title(f"{label}\n{n_closed} closed, {n_open} open")
            
    #         ax.axis("off")
        
    #     # Add colorbar
    #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    #     sm.set_array([])
    #     cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', 
    #                     shrink=0.6, pad=0.1, aspect=30)
    #     cbar.set_label('Switch State (0=Open, 1=Closed)', fontsize=10)
        
    #     plt.tight_layout()
        
    #     if save_path:
    #         plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #         plt.close(fig)
    #     else:
    #         plt.show()


def collapse_switches_to_one_per_line(net):
    """Legacy function for compatibility - now handled in NetworkSwitchManager"""
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
    """Updated to use NetworkSwitchManager with all-lines approach"""
    if not isinstance(switch_probs, torch.Tensor):
        switch_probs = torch.tensor(switch_probs, dtype=torch.float32, device=device)

    # Build edge index from ALL lines (matching new approach)
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
    
    # Now the lengths should match!
    if len(switch_probs) != expected:
        print("🔥  DEBUG: Switch count still mismatched! 🔥")
        print(f"  GNN predicted len:  {len(switch_probs)}")
        print(f"  Expected switches:  {expected}")
        raise ValueError(
            f"Switch count mismatch: GNN predicted {len(switch_probs)}, "
            f"network has {expected}"
        )

    from model_search.models.AdvancedMLP.AdvancedMLP import PhysicsInformedRounding
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
        self.class_0_predictions = []
        self.class_1_predictions = []

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
        print(f"Predicting on {len(loader.dataset)} samples with {self.eval_args.model_module} model…")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader, desc="Predicting", leave=False)):
                start = time.time()
                batch = batch.to(self.device)
                out = self.model(batch)
                logits = out.get("switch_logits")
                if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
                    prob_closed = torch.sigmoid(logits).squeeze(-1)
                    prob_open = 1.0 - prob_closed
                    probs = torch.stack([prob_open, prob_closed], dim=-1)  
                elif logits.dim() >= 2 and logits.size(-1) == 2:
                    # 2
                    probs = torch.softmax(logits, dim=-1)  
                else:
                    raise ValueError(f"Unexpected logits shape {tuple(logits.shape)}")
                

                gnn_time = time.time() - start
                probs_list = probs.cpu().numpy().tolist()
                self.class_0_predictions.extend([p[0] for p in probs_list])
                self.class_1_predictions.extend([p[1] for p in probs_list])

                probs = probs.cpu().numpy().tolist()
                
                gid =  batch.graph_id[0]
                preds[gid] = probs_list
                self.gnn_times[gid] = gnn_time
            return preds, self.gnn_times



class WarmstartSOCP(SOCP_class):
    def __init__(self, net, graph_id="", **kwargs):
        # Extract our custom parameters before calling super().__init__()
        self.fixed_switches = kwargs.pop('fixed_switches', {})
        self.float_warmstart = kwargs.pop('float_warmstart', None)
        
        # Ensure fixed_switches is always a dict, never None
        if self.fixed_switches is None:
            self.fixed_switches = {}
        
        # Now call parent constructor with remaining kwargs
        super().__init__(net=net, graph_id=graph_id, **kwargs)

    def initialize(self):
        super().initialize()
        # Apply fixed switches if any
        if self.fixed_switches:  # Check if dict is not empty
            for idx, val in self.fixed_switches.items():
                if idx in self.switch_df.index:  # Check if index exists
                    self.switch_df.loc[idx, 'closed'] = bool(val)
    
    def create_model(self):
        model = super().create_model()
        # Fix switches in the optimization model
        if self.fixed_switches:  # Check if dict is not empty
            for idx, val in self.fixed_switches.items():
                if idx in self.switch_df.index:  # Check if index exists
                    row = self.switch_df.loc[idx]
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

        opt = SolverFactory(solver)

        if solver == "gurobi_persistent" and self.float_warmstart is not None:
            if hasattr(opt, 'set_instance'):
                opt.set_instance(m)
            
            sorted_switches = self.switch_df[self.switch_df['et'] == 'l'].sort_values('element')
            unique_switches = sorted_switches.drop_duplicates(subset='element', keep='first')
            
            warmstart_count = 0
            gurobi_hints_set = 0
            
            for i, (_, switch_row) in enumerate(unique_switches.iterrows()):
                if i < len(self.float_warmstart):
                    line_id = switch_row['element']
                    if hasattr(m, 'line_status') and line_id in m.line_status:
                        warmstart_value = float(self.float_warmstart[i])
                        warmstart_value = max(0.0, min(1.0, warmstart_value))
        
                        # For Gurobi, use round to nearest integer for binary vars
                        binary_value = 1 if warmstart_value >= 0.5 else 0
                        
                        m.line_status[line_id].set_value(binary_value, skip_validation=True)
                        warmstart_count += 1
                        
                        if hasattr(opt, '_solver_model'):
                            try:
                                pyomo_var = m.line_status[line_id]
                                if hasattr(opt, '_pyomo_var_to_solver_var_map'):
                                    gurobi_var = opt._pyomo_var_to_solver_var_map.get(pyomo_var)
                                    if gurobi_var is not None:
                                        gurobi_var.VarHintVal = warmstart_value
                                        gurobi_hints_set += 1
                                elif hasattr(opt, '_solver_model') and hasattr(opt._solver_model, 'getVars'):
                                    for gvar in opt._solver_model.getVars():
                                        if f"line_status[{line_id}]" in str(gvar.VarName):
                                            gvar.VarHintVal = warmstart_value
                                            gurobi_hints_set += 1
                                            break
                            except Exception as hint_error:
                                pass
                        
                        if i < 5 or i % 20 == 0:
                            print(f"Set warmstart for line {line_id}: binary={binary_value}, hint={warmstart_value:.3f}")
            
            print(f"Float warmstart summary: {warmstart_count} binary values set, {gurobi_hints_set} Gurobi hints set")
        
        elif self.float_warmstart is not None:
            unique_switches = self.switch_df[self.switch_df['et'] == 'l'].sort_values('element').drop_duplicates(subset='element', keep='first')
            
            warmstart_count = 0
            for i, (_, switch_row) in enumerate(unique_switches.iterrows()):
                if i < len(self.float_warmstart):
                    line_id = switch_row['element']
                    if hasattr(m, 'line_status') and line_id in m.line_status:
                        warmstart_value = float(self.float_warmstart[i])
                        binary_value = 1 if warmstart_value >= 0.5 else 0
                        m.line_status[line_id].set_value(binary_value, skip_validation=True)
                        warmstart_count += 1
                        if i < 5: 
                            print(f"Set binary warmstart for line {line_id}: {binary_value}")
            
            print(f"Binary warmstart summary: {warmstart_count} values set")
        
        # Set solver options
        if hasattr(opt, 'options'):
            opt.options.update(opts)
        
        # Set instance if not already done
        if hasattr(opt, 'set_instance') and not hasattr(opt, '_solver_model'):
            opt.set_instance(m)
        
        print(f"Solving with float warmstart using {len(self.float_warmstart) if self.float_warmstart else 0} warmstart values")
        res = opt.solve(m, tee=False, load_solutions=True)
        self.solve_time = getattr(res.solver, 'time', 0.0)
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


    def apply_rounding(self, predictions_2class: list, method: str, switch_manager: NetworkSwitchManager) -> list:
        """Apply rounding using NetworkSwitchManager with 2-class probabilities"""
        if method == "round":
            # Use argmax: choose class with higher probability
            rounded = [1 if pair[1] > pair[0] else 0 for pair in predictions_2class]
        elif method == "PhyR":
            # Extract closed probabilities for PhyR
            closed_probs = [pair[1] for pair in predictions_2class]
            rounded = apply_physics_informed_rounding(closed_probs, switch_manager, device=self.device)
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
        self.graph_ids = list(self.pp_all["mst"].keys())

        # align predictions/network IDs
        # preds = self.predictions.keys()
        # nets  = self.graph_ids
        # if set(preds) != set(nets):
        #     missing_preds = set(nets) - set(preds)
        #     missing_nets  = set(preds) - set(nets)
        #     if missing_preds:
        #         print(f"Warning: Predictions missing for graphs: {missing_preds}")
        #     if missing_nets:
        #         print(f"Warning: Networks missing for predictions: {missing_nets}")
        #     self.graph_ids = [gid for gid in self.graph_ids if gid in self.predictions]


    def _optimize_single(self, args):
        idx, mode = args
        gid = self.graph_ids[idx]
        print(f"Processing graph {idx+1}/{len(self.graph_ids)}: {gid} with mode '{mode}'")
        gnn_time = self.gnn_times.get(gid, 0.0)


        print(f"retrieving gnn output of {gid} to process optimization of {gid}")
        try:
            # Create network and manager
            net = self.pp_all["mst"][gid].deepcopy()
            switch_manager = NetworkSwitchManager(net)

            print(f"network {gid} has {len(switch_manager.collapsed_switches)} switches and nodes: {len(switch_manager.net.bus)}")
            
            # Store manager for visualization
            self.saver.graph_managers[gid] = switch_manager
            
            # 1. Initial state is already captured in manager constructor
            initial_state = switch_manager.get_initial_states()
            
            # 3. GNN probabilities
            raw_preds_2class = self.predictions[gid]  # Now 2-class probabilities
            # Extract closed probabilities for visualization
            closed_probs = [pair[1] for pair in raw_preds_2class]
            switch_manager.add_state(closed_probs, "gnn_probs")
            
            # 4. GNN rounded predictions
            rounded = self.saver.apply_rounding(raw_preds_2class, self.saver.rounding_method, switch_manager)
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
                # Extract closed probabilities for float warmstart
                closed_probs = [pair[1] for pair in raw_preds_2class]
                solver = WarmstartSOCP(net=net, toggles=self.toggles, graph_id=gid, float_warmstart=closed_probs)
                
                # Also record discrete version for saving
                net2 = net.deepcopy()
                switch_manager2 = NetworkSwitchManager(net2)
                switch_manager2.set_switch_states(rounded, "float_warmstart_discrete")
                self.saver.save_network_as_json(net2, self.saver.warmstart_networks_folder, gid)
                
                print(f"Float warmstart {gid}: using raw predictions as float warmstart")
                print(f"Amount of switches rounded to 0/1: {rounded.count(0)}/{rounded.count(1)}")
                avg_0 = np.mean([p for p in closed_probs if p < 0.5]) if any(p < 0.5 for p in closed_probs) else 0
                avg_1 = np.mean([p for p in closed_probs if p >= 0.5]) if any(p >= 0.5 for p in closed_probs) else 0
                print(f"Average value of switches rounded to 0: {avg_0:.2f}, 1: {avg_1:.2f}")
            
            elif mode == "hard":
                T = self.confidence_threshold
                fixed = {}
                fixed_0_count = 0
                fixed_1_count = 0
                sorted_switches = switch_manager.collapsed_switches.sort_values("element")
                for i, (_, switch_row) in enumerate(sorted_switches.iterrows()):
                    if i >= len(raw_preds_2class): break
                    prob_open, prob_closed = raw_preds_2class[i]
                    line_id = switch_row['element']

                    if prob_open >= T:
                        d = 0
                        fixed_0_count += 1
                    elif prob_closed >= T:
                        d = 1
                        fixed_1_count += 1
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
                solver.solve(solver="gurobi_persistent", TimeLimit=6000, MIPGap=1e-2, threads=8)
                solve_time = solver.solve_time
                objective_value = float(pyo_val(solver.model.objective))
                
                # Extract solution and apply to network
                final_net = self.pp_all["mst"][gid].deepcopy()
                final_switch_manager = NetworkSwitchManager(final_net)
                
                sol = {l: round(pyo_val(solver.model.line_status [l])) for l in solver.model.lines}
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
                initial_state=initial_state,
                gnn_probs=closed_probs,  # Use extracted closed probabilities
                gnn_prediction=rounded,
                warmstart_config=rounded,
                final_optima=final_states,
                gnn_time=gnn_time,
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
        
            row = self.saver.csv_data[-1]   
            return gid, {'success': True, 'row': row}

        except Exception as e:
            print(f"ERROR in graph {gid}: {str(e)}") 
            self.saver.add_csv_entry(
                graph_id=gid,
                ground_truth=[],
                initial_state=[],
                gnn_probs=closed_probs,
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
            err_row = {
                "graph_id": gid,
                "error": str(e)
            }
            return gid, {'success': False, 'error': str(e), 'row': err_row}

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
                    if 'row' in res:
                        self.saver.csv_data.append(res['row'])
                    else:
                        err_row = {
                            "graph_id": gid,
                            "error": res['error']
                        }
                        self.saver.csv_data.append(err_row)

 
        else:
            for gid, res in map(self._optimize_single, args):
                results[gid] = res
                if 'row' in res:
                    self.saver.csv_data.append(res['row'])
                else:
                    err_row = { "graph_id": gid, "error": res['error'] }
                    self.saver.csv_data.append(err_row)

        csv_path = self.saver.save_csv()
        return results, csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict-then-Optimize Pipeline with Explicit Saving")
    parser.add_argument("--config_path", type=str,
                        default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\model_search\models\AdvancedMLP\config_files\AdvancedMLP------devout-glitter-19.yaml",
                        help="Path to the YAML config file")
    parser.add_argument("--model_path", type=str,
                        default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\model_search\models\AdvancedMLP\devout-glitter-19-Best.pt", 
                        help="Path to pretrained GNN checkpoint")
    parser.add_argument("--folder_names", type=str, nargs="+",
                        default=[r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\split_datasets\test"],
                        #default = [r"data/split_datasets/test"],
                        help="Folder containing 'mst' and 'mst_opt' subfolders")
    parser.add_argument("--dataset_names", type=str, nargs="+",
                        default=["test"],
                        help="Dataset names corresponding to folder_names")
    
    # Warmstart arguments
    parser.add_argument("--warmstart_mode", type=str,
                        choices=["only_gnn_predictions", "soft", "float", "hard", "optimization_without_warmstart"],
                        default="float",
                        help="Warmstart strategy")
    parser.add_argument("--rounding_method", type=str,
                        choices=["round", "PhyR"],
                        default="PhyR",
                        help="Rounding method for predictions")
    parser.add_argument("--confidence_threshold", type=float, default=0.99,
                        help="Confidence threshold for hard warmstart")
    
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of CPU workers for optimization")
    parser.add_argument("--predict", action="store_true", default=True, 
                        help="Run prediction step")
    parser.add_argument("--optimize", action="store_true", default=True, 
                        help="Run optimization step")
    parser.add_argument("--visualize", default=True, help="Save visualizations of all graphs")
    
    args = parser.parse_args()

    # Build DataLoaders
    from load_data import create_data_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pp_networks = load_pp_networks(args.folder_names[0])

    graph_ids = sorted(pp_networks["mst"].keys())

    loaders = create_data_loaders(
        dataset_names=args.dataset_names,
        folder_names=args.folder_names,
        dataset_type="default",
        shuffle=False,
        batch_size=1)
    test_loader = loaders.get("test", None)

    # Verify alignment
    if len(test_loader.dataset) != len(graph_ids):
        print(f"Warning: Dataset size ({len(test_loader.dataset)}) != number of graphs ({len(graph_ids)})")
    
    # print(f"Test loader created with {len(test_loader.dataset)} samples")
    # print(f"Graph IDs: {graph_ids[:5]}..." if len(graph_ids) > 5 else f"Graph IDs: {graph_ids}")

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

        # print(f"Prediction keys: {sorted(predictions.keys())}")
        # print(f"Graph IDs: {sorted(graph_ids)}")
        # print(f"Predictions type: {type(list(predictions.keys())[0]) if predictions else 'None'}")
        # print(f"Graph IDs type: {type(graph_ids[0]) if graph_ids else 'None'}")
        # # Convert to numpy arrays
        # class_0_preds = np.array(predictor.class_0_predictions)
        # class_1_preds = np.array(predictor.class_1_predictions)

        # class_0_high_conf = class_0_preds[class_0_preds > 0.5]
        # class_1_high_conf = class_1_preds[class_1_preds > 0.5]

        # conf0 = class_0_preds[class_0_preds > 0.5]
        # conf1 = class_1_preds[class_1_preds > 0.5]

        # # create 50 bins in [0.5, 1.0]
        # bins = np.linspace(0.5, 1.0, 51)

        # plt.rcParams.update({
        # 'font.size':         14,
        # 'axes.titlesize':    16,
        # 'axes.labelsize':    14,
        # 'xtick.labelsize':   12,
        # 'ytick.labelsize':   12,
        # 'legend.fontsize':   12,
        # 'figure.titlesize':  16,
        # 'colorbar.labelsize':12,
        #     })

        # # ── Your existing code ──────────────────────────────────────
        # fig, ax0 = plt.subplots(figsize=(10, 5))
        # ax1 = ax0.twinx()

        # # histograms
        # ax0.hist(conf0, bins=bins, alpha=0.5, edgecolor='red', label='Class 0')
        # ax1.hist(conf1, bins=bins, alpha=0.5, edgecolor='blue', label='Class 1')

        # # ── Option B: per‐call overrides ─────────────────────────────
        # ax0.set_xlabel('Confidence',                   fontsize=14)
        # ax0.set_ylabel('Frequency (Class 0)', color='red', fontsize=14)
        # ax1.set_ylabel('Frequency (Class 1)', color='blue', fontsize=14)

        # ax0.set_title('High-Confidence Predictions (> 0.5)', fontsize=16)
        # ax0.tick_params(axis='both', labelsize=12)

        # # custom legend font size
        # h0, l0 = ax0.get_legend_handles_labels()
        # h1, l1 = ax1.get_legend_handles_labels()
        # leg = ax0.legend(h0 + h1, l0 + l1, loc='upper center')
        # for text in leg.get_texts():
        #     text.set_fontsize(12)

        # plt.tight_layout()
        # #plt.show()
        # plt.savefig(saver.predictions_folder / f"confidence_histograms.png")
                            
        
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
        
        # if args.visualize:
        #     # Save visualizations for all graphs
        #     print("\nSaving visualizations for all graphs...")
        #     saver.save_all_visualizations()
        #     print(f"Visualizations saved to: {saver.visualization_folder}")

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