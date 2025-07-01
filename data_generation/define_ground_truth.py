import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pandapower as pp
import numpy as np
import os
import sys
import time
from pathlib import Path
import argparse
import networkx as nx
import logging
import matplotlib.pyplot as plt
import traceback
import copy 
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pandapower.powerflow import LoadflowNotConverged
import json 
from pandapower import from_json_dict
# Add necessary source paths
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
load_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_search"))
if src_path not in sys.path:
    sys.path.append(src_path)
if load_data_path not in sys.path:
    sys.path.append(load_data_path)

from SOCP_class_dnr import SOCP_class
from optimization_logging import setup_logging, get_logger
from pyomo.environ import (value as pyo_val)

SHARED_LOG_PATH = Path(__file__).parent / "logs"/ "define_ground_truth.log"

def init_application_logging(debug=True):
    """Initialize application-wide logging."""
    log_level = logging.INFO if debug else logging.INFO
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "define_ground_truth.log"
    return setup_logging(log_level=log_level, log_file=log_file)

def init_worker_logging():
    """Initialize logging for worker processes."""
    log_file = Path(__file__).parent / "logs" / "define_ground_truth.log"
    setup_logging(log_level=logging.INFO, log_file=log_file)

def get_n_workers():
    workers =int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"Number of workers: {workers}")
    return workers

# def load_graph_data_old(base_directory):
#     logger.info("Loading stored data from %s", base_directory)

#     # Load features
#     features = {}
#     features_dir = os.path.join(base_directory, "graph_features")
#     if os.path.isdir(features_dir):
#         for fn in os.listdir(features_dir):
#             if not fn.endswith(".pkl"): 
#                 continue
#             key = fn[:-4]
#             path = os.path.join(features_dir, fn)
#             logger.debug("  → loading feature %s", key)
#             with open(path, "rb") as f:
#                 features[key] = pkl.load(f)
#     else:
#         logger.warning("No graph_features folder at %s", features_dir)
#     logger.info("Loaded %d feature sets", len(features))

#     # Load NetworkX
#     nx_graphs = {}
#     nx_dir = os.path.join(base_directory, "networkx_graphs")
#     if os.path.isdir(nx_dir):
#         for fn in os.listdir(nx_dir):
#             if not fn.endswith(".pkl"): 
#                 continue
#             key = fn[:-4]
#             path = os.path.join(nx_dir, fn)
#             try:
#                 with open(path, "rb") as f:
#                     nx_graphs[key] = pkl.load(f)
#             except Exception as e:
#                 logger.error("Failed loading NX graph %s: %s", key, e)
#     else:
#         logger.warning("No networkx_graphs folder at %s", nx_dir)
#     logger.info("Loaded %d NetworkX graphs", len(nx_graphs))

#     # Load pandapower
#     pp_networks = {}
#     pp_dir = os.path.join(base_directory, "pandapower_networks")
#     if os.path.isdir(pp_dir):
#         for fn in os.listdir(pp_dir):
#             if not fn.endswith(".json"): 
#                 continue
#             key = fn[:-5]
#             path = os.path.join(pp_dir, fn)
#             try:
#                 with open(path) as f:
#                     raw = f.read()
#                 try:
#                     pp_networks[key] = pp.from_json_string(raw)
#                 except Exception:
#                     pp_networks[key] = json.loads(raw)
#                 logger.debug("  → loaded pandapower network %s", key)
#             except Exception as e:
#                 logger.error("Failed loading pandapower %s: %s", key, e)
#     else:
#         logger.warning("No pandapower_networks folder at %s", pp_dir)
#     logger.info("Loaded %d Pandapower networks", len(pp_networks))

#     return nx_graphs, pp_networks, features
def load_pp_networks(base_directory):
    nets = {}
    folder = os.path.join(base_directory, "original", "pandapower_networks")
    for fn in tqdm(os.listdir(folder), desc=f"Loading original networks from {folder}"):
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
        nets[fn] = net
    return nets

def build_nx_graph(net, include_switches=False, include_trafos=False):
    G = nx.Graph()
    G.add_nodes_from(net.bus.index)
    
    # Always start with all lines 
    for idx, ln in net.line.iterrows():
        G.add_edge(int(ln.from_bus), int(ln.to_bus),
                   type='line', closed=True, line_id=int(idx))
    
    if include_switches:
        # Replace line connections with switch representations
        for s, sw in net.switch.iterrows():
            if sw.et == 'l':  
                # Get the line this switch controls
                line_idx = sw.element
                if line_idx in net.line.index:
                    ln = net.line.loc[line_idx]
                    fb, tb = int(ln.from_bus), int(ln.to_bus)
                    
                    # Remove the original line edge
                    if G.has_edge(fb, tb):
                        G.remove_edge(fb, tb)
                    
                    # Add the switch edge with its state
                    G.add_edge(fb, tb, 
                              type='switch', 
                              closed=bool(sw.closed),
                              switch_id=int(s),
                              line_id=int(line_idx))
    
    if include_trafos:
        # Add transformer connections
        for idx, tr in net.trafo.iterrows():
            if idx in net.trafo.index:  
                G.add_edge(int(tr.hv_bus), int(tr.lv_bus), 
                          type='trafo', trafo_id=int(idx))
        
        # Add 3-winding transformer connections
        for idx, tr in net.trafo3w.iterrows():
            if idx in net.trafo3w.index:  
                G.add_edge(int(tr.hv_bus), int(tr.lv_bus), 
                          type='trafo3w', trafo3w_id=int(idx))
                G.add_edge(int(tr.hv_bus), int(tr.mv_bus), 
                          type='trafo3w', trafo3w_id=int(idx))  
                G.add_edge(int(tr.lv_bus), int(tr.mv_bus), 
                          type='trafo3w', trafo3w_id=int(idx))
    
    return G
def plot_grid_component(net, ax):
    # Build operational graph
    G_operational = build_nx_graph(net, include_switches=True)

    # Build full switch graph 
    G_full = nx.Graph()
    G_full.add_nodes_from(net.bus.index)
    
    # Add all lines first
    for idx, ln in net.line.iterrows():
        G_full.add_edge(int(ln.from_bus), int(ln.to_bus),
                       type='line', closed=True, line_id=int(idx))

    for s, sw in net.switch.iterrows():
        if sw.et == 'l':  
            line_idx = sw.element
            if line_idx in net.line.index:
                ln = net.line.loc[line_idx]
                fb, tb = int(ln.from_bus), int(ln.to_bus)
                
                # Remove the original line edge
                if G_full.has_edge(fb, tb):
                    G_full.remove_edge(fb, tb)
                
                # Add switch edge
                G_full.add_edge(fb, tb, 
                              type='switch', 
                              closed=bool(sw.closed),
                              switch_id=int(s),
                              line_id=int(line_idx))
    
    # Get positions
    if not net.bus_geodata.empty:
        pos = {int(b): (r.x, r.y) for b, r in net.bus_geodata.iterrows()}
    else:
        pos = nx.kamada_kawai_layout(G_full)
    components = list(nx.connected_components(G_operational))
    components = sorted(components, key=len, reverse=True)
    
    import matplotlib.cm as cm
 
    if len(components) > 1:
        colors = cm.tab10(np.arange(len(components)) % 10)
    else:
        colors = ['lightblue']
    
    # Create component color mapping
    node_colors = {}
    component_colors = {}
    for i, component in enumerate(components):
        color = colors[i] if len(components) > 1 else colors[0]
        component_colors[i] = color
        for node in component:
            node_colors[node] = color
    
    # Separate edges by type and state
    regular_edges = []
    closed_switch_edges = []
    open_switch_edges = []
    
    for u, v, data in G_full.edges(data=True):
        edge_type = data.get('type', 'line')
        if edge_type == 'switch':
            if data.get('closed', True):
                closed_switch_edges.append((u, v))
            else:
                open_switch_edges.append((u, v))
        else:
            regular_edges.append((u, v))

    # Draw regular edges colored by component
    for u, v in regular_edges:
        # Determine component color
        edge_color = node_colors.get(u, 'lightgrey')
        nx.draw_networkx_edges(G_full, pos, edgelist=[(u, v)], 
                              edge_color=[edge_color], ax=ax, width=2.0, alpha=0.8)
    
    # Draw closed switches in component colors
    for u, v in closed_switch_edges:
        edge_color = node_colors.get(u, 'lightgrey')
        nx.draw_networkx_edges(G_full, pos, edgelist=[(u, v)], 
                              edge_color=[edge_color], ax=ax, width=2.0, alpha=0.8)
    
    # Draw open switches as dotted black lines
    if open_switch_edges:
        nx.draw_networkx_edges(G_full, pos, edgelist=open_switch_edges, 
                              edge_color="black", ax=ax, width=1.5, 
                              style='dotted', alpha=0.9)

    # Draw nodes colored by component
    for i, component in enumerate(components):
        component_nodes = list(component)
        if component_nodes:
            nx.draw_networkx_nodes(G_full, pos, nodelist=component_nodes,
                                  node_size=40, node_color=[component_colors[i]], 
                                  edgecolors="black", linewidths=0.5, ax=ax)

    ax.axis("off")

    # Add legend for components if multiple
    if len(components) > 1:
        legend_elements = []
        for i, comp in enumerate(components):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=component_colors[i], 
                                            markersize=8, alpha=0.8,
                                            label=f'Comp {i+1} ({len(comp)} buses)'))
        
        # Add open switch legend only if there are open switches
        if open_switch_edges:
            legend_elements.append(plt.Line2D([0], [0], color='black', 
                                            linestyle='dotted', linewidth=1.5,
                                            label='Open Switches'))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                 frameon=True, fancybox=True, shadow=True)

def plot_grid(net, ax, title="", logger=None):
    G = build_nx_graph(net,include_switches=True)
    G_top = build_nx_graph(net, include_switches=False, include_trafos=True)

    if not net.bus_geodata.empty:
        pos = {int(b): (r.x, r.y) for b, r in net.bus_geodata.iterrows()}
    else:
        pos = nx.kamada_kawai_layout(G)

    # Find disconnected buses and cycles
    disconnected_buses = find_disconnected_buses(G)
    disconnected_buses_ful_top = find_disconnected_buses(G_top)

    cycle_edges = find_cycles(G)
    logger.info(f"Disconnected buses: {disconnected_buses}")
    logger.info(f"Disconnected buses in full topology: {disconnected_buses_ful_top}")   
    logger.info(f"Cycle edges: {cycle_edges}")
    # Separate edges by type
    regular_edges = []
    open_edges = []
    cycle_edges_list = []
    
    for u, v, data in G.edges(data=True):
        edge_tuple = (min(u,v), max(u,v))
        if 'closed' in data and not data['closed']:
            open_edges.append((u, v))
        elif edge_tuple in cycle_edges:
            cycle_edges_list.append((u, v))
        else:
            regular_edges.append((u, v))

    # all edges / nodes
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, edge_color="lightgrey", ax=ax, width=1.0)
    nx.draw_networkx_nodes(G, pos, node_size=25, node_color="lightgrey", edgecolors="lightgrey", ax=ax)
    
    if cycle_edges_list:
        nx.draw_networkx_edges(G, pos, edgelist=cycle_edges_list,edge_color="blue", ax=ax, width=0.1, alpha=0.8)
        cycle_nodes = {n for edge in cycle_edges_list for n in edge}
        nx.draw_networkx_nodes(G, pos, nodelist=list(cycle_nodes),
                              node_size=10, node_color="blue", ax=ax)

    
    # open switches
    open_edges = [(u, v) for u, v, d in G.edges(data=True)
              if not d.get("closed", True)]
    if open_edges:
        nx.draw_networkx_edges(G, pos, edgelist=open_edges, edge_color="limegreen", ax=ax, width=2.0)
        nx.draw_networkx_edges(G, pos, edgelist=open_edges,
                               edge_color="limegreen", width=2.0, ax=ax)
        incident = {n for e in open_edges for n in e}
        nx.draw_networkx_nodes(G, pos, nodelist=list(incident),
                               node_size=35, node_color="limegreen",
                               edgecolors="black", linewidths=1.5, ax=ax)
    if disconnected_buses in disconnected_buses_ful_top:
        # Draw disconnected buses
        nx.draw_networkx_nodes(G, pos, nodelist=list(disconnected_buses_ful_top),
                               node_size=50, node_color="red",
                               edgecolors="red", linewidths=1, ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, nodelist=list(disconnected_buses), node_size=50,
                            node_color="yellow", edgecolors="yellow",node_shape="*", linewidths=1, ax=ax)

    ax.set_title(title+ "open switches: " + str(len(open_edges)))
    subtitle = f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
    ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, ha="center", va="top", 
            fontsize=8, style='italic', color='gray')
    ax.axis("off")


def plot_voltage_profile(voltage_data, ax, bus_labels=None, title="Voltage profile"):
    voltage_data = np.asarray(voltage_data, dtype=float)
    
    num_phases = 0
    if voltage_data.ndim > 1:
        num_phases = voltage_data.shape[1]
    elif voltage_data.ndim == 1 and voltage_data.size > 0 :
        num_phases = 1 
        voltage_data = voltage_data[:, np.newaxis] 
    else: 
        ax.text(0.5, 0.5, "No voltage data available", ha='center', va='center')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Bus index / Label")
        ax.set_ylabel("Voltage (p.u.)")
        return

    default_labels = ["Original", "Processed", "MST", "SOCP Optimized", "Original Optimized"]
    phase_labels = default_labels[:num_phases]
    if num_phases > len(default_labels):
        phase_labels.extend([f"Snapshot {i+1}" for i in range(len(default_labels), num_phases)])

    x_ticks = np.arange(voltage_data.shape[0])
    for col in range(voltage_data.shape[1]):
        ax.plot(x_ticks, voltage_data[:, col], label=phase_labels[col], marker='.', linestyle='-', markersize=4, linewidth=1.2)

    ax.axhline(1.05, linestyle="--", linewidth=0.8, color="grey")
    ax.axhline(0.95, linestyle="--", linewidth=0.8, color="grey")
    
    if bus_labels is not None and len(bus_labels) == len(x_ticks):
        if len(x_ticks) > 20: 
             step = len(x_ticks) // 10
             ax.set_xticks(x_ticks[::step])
             ax.set_xticklabels(bus_labels[::step], rotation=45, ha="right")
        else:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(bus_labels, rotation=45, ha="right")
    else:
        ax.set_xlabel("Bus index")

    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if voltage_data.shape[1] > 1 or len(phase_labels) == 1 : 
        ax.legend(frameon=False)


def find_disconnected_buses(G):
    G_closed = nx.Graph()
    G_closed.add_nodes_from(G.nodes())
    
    # Only include closed edges
    for u, v, data in G.edges(data=True):
        if data.get('closed', True):
            G_closed.add_edge(u, v)
    
    disconnected_buses = set()
    
    for component in nx.connected_components(G_closed):
        if len(component) <= 2:
            disconnected_buses.update(component)
        for node in component:
            if G_closed.degree(node) == 0:
                disconnected_buses.add(node)
    
    return disconnected_buses


def loss_improvement(net_before, net_after, include_trafos=True,logger=None):
    pp.runpp(net_before, enforce_q_lims=False)
    pp.runpp(net_after,  enforce_q_lims=False)

    def active_loss(net):
        loss = net.res_line["pl_mw"].abs().sum()
        #if include_trafos and "res_trafo" in net:
        #    loss += net.res_trafo["pl_mw"].abs().sum()

        return float(loss)

    l0 = active_loss(net_before)
    l1 = active_loss(net_after)

    def manual_loss(net):
        manual_loss =[ ]
        for idx, ln in net.line.iterrows():
            R_tot = ln.r_ohm_per_km * ln.length_km
            I_amp = net.res_line.at[idx, "i_ka"] * 1000
            p_loss_mw = (R_tot * I_amp**2) / 1e6
            manual_loss.append(p_loss_mw)
        return sum(manual_loss)
    l0 = manual_loss(net_before)
    l1 = manual_loss(net_after)
    return {
        "loss_before":       l0,
        "loss_after":        l1,
        "loss_improvement":  100.0 * (l0 - l1) / l0 if l0 > 0 else np.nan
    }


def count_switch_changes(net_a, net_b):
    sa = net_a.switch[(net_a.switch.et=='l')]['closed']
    sb = net_b.switch[(net_b.switch.et=='l')]['closed']
    common = sa.index.intersection(sb.index)
    return int((sa.loc[common] != sb.loc[common]).sum())

def find_cycles(G):
    G_closed = nx.Graph()
    G_closed.add_nodes_from(G.nodes())

    # Add only closed edges (for cycle detection)
    for u, v, data in G.edges(data=True):
        if data.get('closed', True):  
            G_closed.add_edge(u, v, **data)
    cycle_edges = set()
    
    # Check each connected component for cycles
    for component in nx.connected_components(G_closed):
        if len(component) <= 2:
            continue
            
        subgraph = G_closed.subgraph(component)
        if not nx.is_tree(subgraph):
            # Find all cycles in this component
            try:
                cycles = nx.cycle_basis(subgraph)
                for cycle in cycles:
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        cycle_edges.add((min(u, v), max(u, v)))
            except:
                for u, v in subgraph.edges():
                    cycle_edges.add((min(u, v), max(u, v)))
    
    return cycle_edges


def is_radial_and_connected(net, y_mask=None, include_switches=False, include_trafos=False):
    G = build_nx_graph(net, include_switches=include_switches, include_trafos=include_trafos)
    
    if y_mask is not None:
        if isinstance(y_mask, pd.Series):
            active_buses = set(net.bus.index[y_mask])
        else:
            active_buses = set(y_mask)
        
        nodes_to_remove = [n for n in G.nodes() if n not in active_buses]
        G.remove_nodes_from(nodes_to_remove)
    
    cycles = find_cycles(G)
    disconnected_buses = find_disconnected_buses(G)
    
    radial = len(cycles) == 0
    connected = len(disconnected_buses) == 0
    
    return radial, connected

def visualize_network_states(snapshots, graph_id, output_dir=None, debug=False, logger=None):   
    if logger is None:
        logger = get_logger(SHARED_LOG_PATH, debug=debug)

    logger.info("printing_network_states")
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    nets_to_visualize = []
    labels = []
    
    snapshot_keys_ordered = ["original", "processed", "mst", "optimised_mst"]
    if debug:
        snapshot_keys_ordered.append("optimised_orig")

    for key in snapshot_keys_ordered:
        if key in snapshots and snapshots[key] is not None:
            nets_to_visualize.append(snapshots[key])
            labels.append(key.replace("_", " ").title())
    n_nets = len(nets_to_visualize)
    
    if n_nets == 0:
        print(f"No networks to visualize for graph {graph_id}")
        return
    
    fig, axes = plt.subplots(2, n_nets + 1, figsize=(4 * (n_nets + 1), 10))  
    
    if n_nets + 1 == 1:
        axes = axes.reshape(2, 1)

    common_ref_bus = None
    processed_snapshot = snapshots.get("processed") 
    
    if processed_snapshot and hasattr(processed_snapshot, 'ext_grid') and not processed_snapshot.ext_grid.empty:
        common_ref_bus = processed_snapshot.ext_grid.bus.iloc[0]
        logger.info(f"Primary common reference bus for voltage profiles: {common_ref_bus} (from processed_net ext_grid).")
    else:
        logger.warning("Could not determine common_ref_bus from 'processed' network. Voltage profiles might be inconsistent if 'processed' net is missing or has no slack.")

    all_bus_indices = set()
    for net_key in snapshot_keys_ordered: 
        net_val = snapshots.get(net_key)
        if net_val and hasattr(net_val, "bus"):
            all_bus_indices.update(net_val.bus.index.tolist())
    reference_bus_index = pd.Index(sorted(list(all_bus_indices)))
    if not reference_bus_index.empty:
        logger.debug(f"Voltage profile x-axis (reference_bus_index) created with {len(reference_bus_index)} buses.")
    else:
        logger.warning("Reference bus index for voltage profile is empty. Plot may not generate correctly.")
    aligned_voltage_data = []
    for i, net_orig in enumerate(nets_to_visualize):
        label = labels[i]
        net = copy.deepcopy(net_orig)
        
        logger.debug(f"Processing snapshot '{label}' for voltage profile.")

        try:
            if common_ref_bus is not None and common_ref_bus in net.bus.index:
                if hasattr(net, 'ext_grid'):
                    net.ext_grid = net.ext_grid[0:0] 

                pp.create_ext_grid(net, bus=common_ref_bus, vm_pu=1.0, va_degree=0.0, name=f"Vis_Ref_{label}")
                logger.info(f"Temporarily set ext_grid at bus {common_ref_bus} for '{label}' for consistent voltage profile PF.")
                
                pp.runpp(net, enforce_q_lims=False, calculate_voltage_angles=True, algorithm='nr')
            
            elif common_ref_bus is None:
                logger.warning(f"No common_ref_bus determined. Running PF for '{label}' with its original setup.")
                if not (hasattr(net, 'res_bus') and 'vm_pu' in net.res_bus.columns and not net.res_bus.empty):
                    pp.runpp(net, enforce_q_lims=False, calculate_voltage_angles=True, algorithm='nr')

            else: 
                logger.warning(f"Common reference bus {common_ref_bus} not in '{label}'. Running PF for '{label}' with its original setup.")
                if not (hasattr(net, 'res_bus') and 'vm_pu' in net.res_bus.columns and not net.res_bus.empty):
                    pp.runpp(net, enforce_q_lims=False, calculate_voltage_angles=True, algorithm='nr')

        except LoadflowNotConverged:
            logger.warning(f"Power flow did NOT converge for '{label}' during voltage profile generation.")
            if not hasattr(net, 'res_bus'): net.res_bus = pd.DataFrame(index=net.bus.index if hasattr(net, 'bus') else None)
            net.res_bus['vm_pu'] = np.nan 
        except Exception as e:
            logger.error(f"Error during power flow for '{label}' in voltage profile generation: {e}")
            logger.debug(traceback.format_exc())
            if not hasattr(net, 'res_bus'): net.res_bus = pd.DataFrame(index=net.bus.index if hasattr(net, 'bus') else None)
            net.res_bus['vm_pu'] = np.nan

        if hasattr(net, 'res_bus') and 'vm_pu' in net.res_bus.columns and not reference_bus_index.empty:
            current_voltages = net.res_bus.vm_pu.reindex(reference_bus_index, fill_value=np.nan)
            aligned_voltage_data.append(current_voltages.values)
        elif not reference_bus_index.empty :
            aligned_voltage_data.append(np.full(len(reference_bus_index), np.nan))
        else: 
             aligned_voltage_data.append(np.array([])) 


        # Plot in both rows
        plot_grid(net, axes[0, i], title=f"{label} - Regular", logger=logger)
        plot_grid_component(net, axes[1, i],)
        
        topo_radial, topo_connected = is_radial_and_connected(net, include_trafos=True)
        op_radial, op_connected = is_radial_and_connected(net, include_switches=True)

        axes[0, i].text(0.5, -0.12,
                    f"Topological Graph: Connected: {topo_connected} Radial: {topo_radial} \nOperational Graph:Connected: {op_connected} Radial: {op_radial}",
                    transform=axes[0, i].transAxes,
                    ha="center", va="top", fontsize=7)
        

    # Plot voltage profile in top right
    voltage_plot_ax = axes[0, n_nets]
    if aligned_voltage_data:
        stacked_voltages = np.column_stack(aligned_voltage_data)
        bus_labels_for_plot = [str(i) for i in reference_bus_index.tolist()]
        plot_voltage_profile(stacked_voltages, voltage_plot_ax, bus_labels=bus_labels_for_plot, title="Voltage Profile")

    axes[1, n_nets].axis("off")
    axes[1, n_nets].set_visible(False)
    
    plt.tight_layout()
    if output_dir:
        filename = output_dir / f"{graph_id}_network_states.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()
def preprocess_network_for_optimization(net, logger=None, debug=False, ):

    # Find distribution voltage level (typically the most common lower voltage)
    if hasattr(net, 'bus') and 'vn_kv' in net.bus.columns:
        voltage_levels = net.bus['vn_kv'].value_counts()
        distribution_voltage = voltage_levels.index[0] 
        logger.info(f"Identified distribution voltage level: {distribution_voltage} kV")
        
        # Only include buses at distribution level
        distribution_buses = net.bus[net.bus['vn_kv'] == distribution_voltage].index
    else:
        distribution_buses = net.bus.index
    
    # Build graph with only distribution lines
    G_dist_only = nx.Graph()
    G_dist_only.add_nodes_from(distribution_buses)
    
    for _, ln in net.line.iterrows():
        if ln.from_bus in distribution_buses and ln.to_bus in distribution_buses:
            G_dist_only.add_edge(int(ln.from_bus), int(ln.to_bus), element=int(ln.name))
    
    # Find main distribution component
    dist_components = list(nx.connected_components(G_dist_only))
    if not dist_components:
        logger.warning(f"No distribution components found.")
        return None, None, None
    
    main_dist_component = set(max(dist_components, key=len))
    logger.info(f" Selected main distribution component with {len(main_dist_component)} buses.")
    
    # Create active bus mask for distribution buses only
    active_bus_mask = pd.Series(False, index=net.bus.index)
    active_bus_mask.loc[list(main_dist_component)] = True
    
    # Identify different types of excluded nodes
    G_full_phys = build_nx_graph(net,include_trafos=True)
    node_to_full_phys_comp_id = {
        node: i for i, comp in enumerate(nx.connected_components(G_full_phys)) for node in comp
    }
    
    main_phys_comp_id = None
    if main_dist_component:
        first_bus_in_main = next(iter(main_dist_component), None)
        if first_bus_in_main is not None:
            main_phys_comp_id = node_to_full_phys_comp_id.get(first_bus_in_main)

    transformer_linked_nodes = set()
    other_excluded_nodes = set()

    for bus_idx in net.bus.index:
        if bus_idx not in main_dist_component:
            bus_full_phys_comp_id = node_to_full_phys_comp_id.get(bus_idx)
            if bus_full_phys_comp_id is not None and bus_full_phys_comp_id == main_phys_comp_id:
                transformer_linked_nodes.add(bus_idx)
            else:
                other_excluded_nodes.add(bus_idx)
    
    # Find slack buses
    slack_buses = set()
    if hasattr(net, 'ext_grid') and not net.ext_grid.empty:
        slack_buses.update(net.ext_grid.bus.tolist())
    if hasattr(net, 'gen') and not net.gen.empty:
        if "slack" in net.gen.columns:
            slack_buses.update(net.gen[net.gen.slack].bus.tolist())
        elif "type" in net.gen.columns:
            slack_buses.update(net.gen[net.gen.type == "slack"].bus.tolist())
    
    # If no slack buses in main component, add one
    if not (main_dist_component & slack_buses):
        logger.warning(f"Selected main component has no original reference buses, adding one...")
        subnet_graph_for_slack = G_dist_only.subgraph(main_dist_component)
        degrees_in_main = dict(subnet_graph_for_slack.degree())
        if degrees_in_main:
            new_slack = max(degrees_in_main, key=degrees_in_main.get)
            slack_buses.add(new_slack)
            logger.warning(f" Added new reference bus {new_slack} to slack_buses set.")

    # Create active bus mask for distribution buses only
    active_bus_mask = pd.Series(False, index=net.bus.index)
    active_bus_mask.loc[list(main_dist_component)] = True
    
    if debug:
        logger.debug(f" Debugging graph connectivity:")
        logger.debug(f" Selected main_dist_component size: {len(main_dist_component)}")
        logger.debug(f" Main_dist_component buses: {sorted(list(main_dist_component))}")
        logger.debug(f" Transformer-linked_nodes: {len(transformer_linked_nodes)}")
        logger.debug(f" Other_excluded_nodes: {len(other_excluded_nodes)}")
        logger.debug(f" All identified slack buses: {slack_buses}")
    
    try:
        processed_net = pp.select_subnet(
            net,
            buses=list(main_dist_component),
            include_switch_buses=True, 
            include_results=True
        )
        
        # Ensure power source in processed network
        has_power_source = False
        if hasattr(processed_net, 'ext_grid') and not processed_net.ext_grid.empty:
            has_power_source = True
        if not has_power_source and hasattr(processed_net, 'gen') and not processed_net.gen.empty:
            if any(g_bus in slack_buses for g_bus in processed_net.gen.bus):
                has_power_source = True

        if not has_power_source:
            logger.warning(f" Processed network has no power sources, adding one...")
            candidate_buses_for_ext_grid = main_dist_component & slack_buses
            chosen_slack_bus_for_ext_grid = next(iter(candidate_buses_for_ext_grid), None)
            
            if not chosen_slack_bus_for_ext_grid: 
                subnet_graph = G_dist_only.subgraph(main_dist_component)
                degrees = dict(subnet_graph.degree())
                if degrees:
                    chosen_slack_bus_for_ext_grid = max(degrees, key=degrees.get)
            
            if chosen_slack_bus_for_ext_grid is not None and chosen_slack_bus_for_ext_grid in processed_net.bus.index:
                pp.create_ext_grid(processed_net, chosen_slack_bus_for_ext_grid, vm_pu=1.0, va_degree=0.0)
                logger.warning(f" Added ext_grid at bus {chosen_slack_bus_for_ext_grid}.")
                slack_buses.add(chosen_slack_bus_for_ext_grid)

    except Exception as e:
        logger.error(f"Error creating subnet: {e}")
        logger.error(traceback.format_exc())
        return None, None, None
    
    logger.info(f"Preprocessed network: {len(processed_net.bus)} of {len(net.bus)} buses kept.")
    return processed_net, active_bus_mask, 

def alternative_mst_reconfigure(net, penalty=1.0, y_mask=None):
    # ---------- prepare mask ------------------------------------------
    if y_mask is None:
        active_bus = pd.Series(1, index=net.bus.index)
    else:                                
        active_bus = pd.Series(y_mask, index=net.bus.index).fillna(0).astype(bool)

    # ---------- collect meta ------------------------------------------
    line_df   = net.line.copy()
    switch_df = net.switch.copy()
    init_stat = switch_df['closed'].copy()
    bus_name  = net.bus['name'].to_dict()

    lines_with_switches = {}
    for s, sw in switch_df.query("et=='l'").iterrows():
        ln_idx = sw.element
        ln_name = net.line.at[ln_idx, 'name']
        lines_with_switches.setdefault(ln_name, []).append(s)

    # ---------- build candidate graph -------------------------------
    G = nx.Graph()
    for b in net.bus.itertuples():
        if active_bus[b.Index]:
            G.add_node(b.name)
    print(f"active buses: {len(G.nodes())}")
    for ln in line_df.itertuples():
        if not (active_bus[ln.from_bus] and active_bus[ln.to_bus]):
            continue                     
        w  = ln.r_ohm_per_km * ln.length_km
        if any(init_stat[s] for s in lines_with_switches.get(ln.name, [])):
            w += penalty                
        G.add_edge(bus_name[ln.from_bus], bus_name[ln.to_bus],
                   weight=w, line_name=ln.name)
    if G.number_of_nodes() < 2:
        for s in switch_df.index:          
            net.switch.at[s, 'closed'] = False
        return net

    # ---------- MST & switch update ----------------------------------
    mst        = nx.minimum_spanning_tree(G)
    mst_edges  = {frozenset(e) for e in mst.edges()}

    for ln_name, sw_list in lines_with_switches.items():
        ln_row = line_df[line_df.name == ln_name].iloc[0]
        fb = bus_name[ln_row.from_bus]
        tb = bus_name[ln_row.to_bus]
        new_state = frozenset((fb, tb)) in mst_edges
        for s in sw_list:
            net.switch.at[s, 'closed'] = new_state

    for s, sw in switch_df.query("et=='l'").iterrows():
        fb = net.line.at[sw.element, 'from_bus']
        tb = net.line.at[sw.element, 'to_bus']
        if not (active_bus[fb] and active_bus[tb]):
            net.switch.at[s, 'closed'] = False

    return net


def apply_socp(net, graph_id, toggles=None, logger=None, active_bus_mask=None, lp_dir=None, debug=False):
    if logger is None:
        logger = get_logger(f"socp.{graph_id}")
    logger.info(f"Applying SOCP optimization to {graph_id}")
    net_before_opt = copy.deepcopy(net)

    # Initialize SOCP optimizer with debug level
    debug_level = 2 if debug else 1  
    optimizer = SOCP_class(net, graph_id, toggles=toggles, active_bus_mask=active_bus_mask, 
                          logger=logger, debug_level=debug_level)
    optimizer.initialize()
    optimizer.model = optimizer.create_model()
    manual_loss_check = sum(    pyo_val(optimizer.model.line_resistance_pu[l]) * pyo_val(optimizer.model.squared_current_magnitude[l, 0]) for l in optimizer.model.lines
    ) * optimizer.S_base_VA / 1e6

    logger.info(f"BEFORE OPT Manual loss check (MW): {manual_loss_check:.6f}")
    logger.info(f"BEFORE OPT SOCP objective loss term (MW): {pyo_val(optimizer.model.loss_term_expr) * optimizer.S_base_VA / 1e6:.6f}")
    
   
    # Save LP file if in debug mode 
    if debug and lp_dir:
        lp_dir = Path(lp_dir)
        lp_dir.mkdir(parents=True, exist_ok=True)
        lp_filename = lp_dir / f"{graph_id}.lp"
        optimizer.model.write(str(lp_filename), io_options={"symbolic_solver_labels": True})
        logger.info(f"LP file saved to {lp_filename}")
    
    # Solve 
    t0 = time.time()
    optimizer.solve()
    opt_time = time.time() - t0

    logger.info(f"net line losses before optimized calculated same as obj:{net}")
    # Update the network with optimization results
    net_opt = optimizer.update_network()
    manual_loss_check = sum(    pyo_val(optimizer.model.line_resistance_pu[l]) * pyo_val(optimizer.model.squared_current_magnitude[l, 0]) for l in optimizer.model.lines
    ) * optimizer.S_base_VA / 1e6

    logger.info(f"Manual loss check (MW): {manual_loss_check:.6f}")
    logger.info(f"SOCP objective loss term (MW): {pyo_val(optimizer.model.loss_term_expr) * optimizer.S_base_VA / 1e6:.6f}")
    # Try to run power flow on the optimized network
    pf_converged = False
    try:
        pp.runpp(net_opt, enforce_q_lims=False)
        pf_converged = net_opt.converged
    except LoadflowNotConverged:
        logger.warning(f"{graph_id}: PF after optimization did not converge")
    
    # Calculate metrics
    flips = count_switch_changes(net_before_opt, net_opt)
    rad_before = is_radial_and_connected(net_before_opt, include_switches=True) 
    rad_after = is_radial_and_connected(net_opt, include_switches=True) 
    loss_impr = None
    if pf_converged:
        loss_impr = loss_improvement(net_before_opt, net_opt, logger)
        logger.info(f"Loss before: {loss_impr['loss_before']:.5f} MW, "f"Loss after: {loss_impr['loss_after']:.5f} MW")
    metrics = {
        "graph_id": graph_id,
        "total_switches": net.switch.shape[0],
        "opt_time": opt_time,
        "switches_changed": flips,
        "loss_improvement": loss_impr["loss_improvement"] if loss_impr else None,
        "before_radial": rad_before[0],
        "before_connected": rad_before[1],
        "after_radial": rad_after[0],
        "after_connected": rad_after[1],
        "pf_converged": pf_converged
    }
    
    # Add debug info to metrics if available
    if debug and hasattr(optimizer, 'constraint_violations') and optimizer.constraint_violations:
        metrics["has_violations"] = True
        metrics["violation_count"] = sum(len(v) for v in optimizer.constraint_violations.values())
 
        for constraint_type, violations in optimizer.constraint_violations.items():
            if violations:
                metrics[f"violations_{constraint_type}"] = len(violations)
                
    return net_opt, metrics

def process_single_graph(graph_id, net_json, folder_path, toggles=None, vis_dir=None, lp_dir=None, logger=None, debug=False):
    if logger is None:
        logger = get_logger(f"worker.{graph_id}")
    logger.info(f"Processing graph {graph_id}")
    
    # 1 load and run network -------------------------------------------------------------
    net_orig = net_json
    try: 
        pp.runpp(net_orig, enforce_q_lims=False)
        if not net_orig.converged:
            logger.warning(f"{graph_id}: PF on original net failed - skip")
            return None
    except LoadflowNotConverged:
        logger.warning(f"{graph_id}: PF on original net failed - skip")
        return None
    
    # 2 Preprocess network -------------------------------------------------------------
    processed_net, active_bus_mask,  = preprocess_network_for_optimization(net_orig, debug=debug, logger =logger)
    try:
        pp.runpp(processed_net, enforce_q_lims=False)
    except LoadflowNotConverged:
        logger.warning(f"{graph_id}: PF on processed network failed - skip")
        return None

    # 3 Check radiality and connectivity ---------------------------------------------
    net_mst = alternative_mst_reconfigure(copy.deepcopy(processed_net), penalty=1.0, y_mask=active_bus_mask)

    try:
        pp.runpp(net_mst, enforce_q_lims=False)
    except LoadflowNotConverged:
        logger.warning(f"{graph_id}: PF on MST network failed - skip")
        return None

    # 4 Apply SOCP optimization --------------------------------------------------------------
    net_opt_mst, metrics_mst = apply_socp(
        net_mst, 
        graph_id + "_mst", 
        toggles=toggles, 
        logger=logger,
        active_bus_mask=active_bus_mask,
        lp_dir=lp_dir,
        debug=debug
    )
    rad_conn_mst = is_radial_and_connected(net_mst, include_switches=True) #y_mask=active_bus_mask)
    flips_orig_to_mst = count_switch_changes(processed_net, net_mst)
    flips_mst_to_opt = count_switch_changes(net_mst, net_opt_mst)

    metrics = {
        "graph_id": graph_id,
        "total_switches": processed_net.switch.shape[0],
        "switches_changed_orig_to_mst": flips_orig_to_mst,
        "switches_changed_mst_to_opt": flips_mst_to_opt,
        "opt_time_mst": metrics_mst["opt_time"],
        "loss_improvement_mst_opt": metrics_mst["loss_improvement"],
        "rad_mst": rad_conn_mst,
        "rad_mst_opt": (metrics_mst["after_radial"], metrics_mst["after_connected"]),
    }

    snapshots = {
        "original": net_orig,
        "processed": processed_net,
        "mst": net_mst,
        "mst_opt": net_opt_mst,
    }
    if debug:
        try:
            # Apply SOCP directly to the processed network 
            net_opt_orig, metrics_orig = apply_socp(
                processed_net,
                graph_id + "_orig",
                toggles=toggles,
                logger=logger,
                active_bus_mask=active_bus_mask,
                lp_dir=lp_dir,
                debug=debug
            )
        
            flips_mst_vs_orig = count_switch_changes(net_opt_mst, net_opt_orig)
            
            metrics.update({
                "opt_time_origPF": metrics_orig["opt_time"],
                "switches_changed_orig_to_opt": metrics_orig["switches_changed"],
                "state_diff_between_opts": flips_mst_vs_orig,
                "loss_improvement_orig_opt": metrics_orig["loss_improvement"],
                "rad_origPF_opt": (metrics_orig["after_radial"], metrics_orig["after_connected"])
            })
            snapshots["original_opt"] = net_opt_orig
            logger.info(
                f"{graph_id}: MST→OPT={flips_mst_to_opt} flips; origPF→OPT={metrics_orig['switches_changed']} flips; diff={flips_mst_vs_orig}"
            )
        except Exception as e:
            logger.warning(f"{graph_id}: Error in original network optimization: {e}")
            logger.debug(traceback.format_exc())
    else:
        logger.info(f"{graph_id}: MST only, MST→OPT={flips_mst_to_opt} flips")

    visualize_network_states(snapshots, graph_id, output_dir=vis_dir, debug=debug, logger=logger)
    logger.info("network states visualized")
    store_snapshots(graph_id, folder_path, logger, **snapshots)
    
    return metrics

def store_snapshots(graph_id: str, root_folder: Path, logger, **nets,):
    for phase, net in nets.items():
        out_dir = root_folder / phase / "pandapower_networks"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{graph_id}.json"         
        with open(fname, "w") as fp:
            fp.write(pp.to_json(net))

        logger.info(f"[{graph_id}] snapshot '{phase}' saved → {fname}")
  
def apply_optimization(folder_path, toggles=None, debug=False, serialize=False):
    folder_path = Path(folder_path)
    root_logger = init_application_logging(debug=debug)
    logger = get_logger("optimization")
    logger.info(f"Starting optimization on {folder_path}")
    logger.info(f"Toggles: {toggles}")

    if not serialize:
        init_worker_logging()

    pp_networks = load_pp_networks(folder_path)
    items = [(gid, net) for gid, net in pp_networks.items()]
    #print(items)
    metrics = []
    vis_dir = (Path("data_generation")  / "logs" / "visualizations").resolve()
    lp_dir = (Path("data_generation") / "logs" / "lp_files").resolve()
    
    vis_dir.mkdir(parents=True, exist_ok=True)
    lp_dir.mkdir(parents=True, exist_ok=True)


    if serialize:
            #--- Sequential execution ---
        for idx, (gid, net_json) in enumerate(items):
            res = process_single_graph(gid, net_json, folder_path , toggles,vis_dir,lp_dir,logger=logger, debug = debug)
            if res: metrics.append(res)
            if idx ==0: 
                break
    else: 
        # --- Parallel execution ---

        workers =  get_n_workers()
        logger.info(f"number of workers used: {workers}")

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker_logging
        ) as executor:
            futures = {
                executor.submit(
                    process_single_graph,
                    gid, net_json, folder_path,
                    toggles,vis_dir,lp_dir,logger=logger, debug=debug
                ): gid
                for gid, net_json in items
            }
            for fut in tqdm(as_completed(futures), total=len(futures)):
                gid = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    logging.getLogger("network_optimizer").error(
                        f"{gid}: failed in worker - skipping", exc_info=True
                    )
                    continue
                if res:
                    metrics.append(res)


    df = pd.DataFrame(metrics)
    df.to_csv(folder_path / "optimization_metrics.csv", index=False)
    print(df)

    # Save optimization metrics.
    metrics_df = pd.DataFrame(metrics)
    metrics_csv = folder_path / "optimization_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n✓ Saved optimization metrics to {metrics_csv}")

    # Summary printing.
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Total graphs processed: {len(pp_networks)}")
    print(f"Successful optimizations: {len(metrics)}")


    if not metrics_df.empty:
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if "time" in col or "loss" in col:
                stat = metrics_df[col].mean()
                print(f"Average {col}: {stat:.4f}")
            else:
                stat = metrics_df[col].sum()
                print(f"Total {col}: {stat}")
    print("="*50)

    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        n = len(numeric_cols)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
        axes = np.atleast_1d(axes).flatten()

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            ax.hist(metrics_df[col].dropna(), bins=10)
            ax.set_title(col.replace('_',' ').title())
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
        for ax in axes[n:]:
            ax.set_visible(False)
        plt.savefig(folder_path / "optimization_histograms.png", bbox_inches='tight', dpi=300)
        plt.tight_layout()
        plt.show()
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ground truth data for power networks using optimization')
    parser.add_argument('--folder_path',
                        default = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\source_datasets\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32\test",

                        type=str, help='Dataset folder path')
    parser.add_argument('--debug', type=bool, default=True, help='Print debug information')
    parser.add_argument('--serialize', type=bool, default=False, help='Serialize the optimization results usefull for debugging')
    # SOCP toggles
    parser.add_argument('--include_voltage_drop_constraint', type=bool, default=True, help="Include voltage drop constraint SOCP")
    parser.add_argument('--include_voltage_bounds_constraint', type=bool, default=True, help="Include voltage bounds constraint SOCP")
    parser.add_argument('--include_power_balance_constraint', type=bool, default=True, help="Include power balance constraint SOCP")
    parser.add_argument('--include_cone_constraint', type=bool,default=True,help="Enables/disables conic cone")
    parser.add_argument('--include_radiality_constraints', type=bool, default=True, help="Include radiality constraints SOCP")
    parser.add_argument('--use_spanning_tree_radiality', type=bool, default=False, help="Use spanning tree radiality SOCP")
    parser.add_argument('--use_root_flow', type=bool, default=True, help="Use Single commodity flow SOCP")
    parser.add_argument('--use_parent_child_radiality', type=bool, default=False, help="Use parent-child radiality SOCP")
    parser.add_argument('--allow_load_shed', type=bool, default=True, help="Allow load shedding in SOCP")
    parser.add_argument('--include_switch_penalty', type=bool, default=True, help="Include switch penalty in objective SOCP")
    parser.add_argument('--all_lines_are_switches', type=bool, default=True, help="Include switch penalty in objective SOCP")
    parser.add_argument("--write_files", action="store_true", help="If set, write out LP/MPS model files; otherwise skip for speed")
    args = parser.parse_args()

    logger = init_application_logging(debug=args.debug)
    
    SOCP_toggles = { 
        "include_voltage_drop_constraint": args.include_voltage_drop_constraint, 
        "include_voltage_bounds_constraint": args.include_voltage_bounds_constraint,   
        "include_power_balance_constraint": args.include_power_balance_constraint,  
        "include_radiality_constraints": args.include_radiality_constraints,
        "use_spanning_tree_radiality": args.use_spanning_tree_radiality,  
        "use_root_flow":args.use_root_flow,
        "use_parent_child_radiality": args.use_parent_child_radiality,
        "allow_load_shed": args.allow_load_shed,    
        "include_switch_penalty": args.include_switch_penalty,
        "all_lines_are_switches": args.all_lines_are_switches,
        "include_cone_constraint": args.include_cone_constraint 
    }

    logger.info("Toggles for optimization:")
    logger.info(SOCP_toggles)
    apply_optimization(args.folder_path, toggles=SOCP_toggles, debug=args.debug, serialize=args.serialize)

    print("\nGround truth generation complete!!!!")

