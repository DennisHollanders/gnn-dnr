import json
import pandas as pd
import pandapower as pp
import pickle as pkl
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
import random
import copy 
import pdb 
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pandapower.powerflow import LoadflowNotConverged

# Add necessary source paths
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
load_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_search"))
if src_path not in sys.path:
    sys.path.append(src_path)
if load_data_path not in sys.path:
    sys.path.append(load_data_path)


from SOCP_class_dnr import SOCP_class
from load_data import load_graph_data_old
from electrify_subgraph import extract_node_features, extract_edge_features
from optimization_logging import setup_logging, get_logger


SHARED_LOG_PATH = Path(__file__).parent / "logs"/ "define_ground_truth.log"

def init_application_logging(method="SOCP", debug=True):
    """Initialize application-wide logging."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{method.upper()}_logs.txt"
    return setup_logging(log_level=log_level, log_file=log_file)

def init_worker_logging():
    """Initialize logging for worker processes."""
    log_file = Path(__file__).parent / "logs" / "define_ground_truth.log"
    setup_logging(log_level=logging.DEBUG, log_file=log_file)

def get_n_workers():
    workers =int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"Number of workers: {workers}")
    return workers

def build_nx_graph(net, include_lines=False, include_trafos=False, include_switches=False):
    G = nx.Graph()
    G.add_nodes_from(net.bus.index)
    
    if include_lines:
        for _, ln in net.line.iterrows():
            G.add_edge(int(ln.from_bus), int(ln.to_bus),
                       type='line', closed=True)
    
    if include_switches:
        # Handle line switches
        for s, sw in net.switch.iterrows():
            if sw.et == 'l':  # Line switch
                ln = net.line.loc[sw.element]
                fb, tb = int(ln.from_bus), int(ln.to_bus)
                # Update existing edge or create new one
                if G.has_edge(fb, tb):
                    G.edges[fb, tb]['closed'] = bool(sw.closed)
                    G.edges[fb, tb]['switch_id'] = int(s)
                else:
                    G.add_edge(fb, tb, type='switch', closed=bool(sw.closed), switch_id=int(s))
            elif sw.et == 'b':  # Bus switch
                fb, tb = int(sw.bus), int(sw.element)
                G.add_edge(fb, tb, type='bus_switch', closed=bool(sw.closed), switch_id=int(s))
    
    if include_trafos:
        for _, tr in net.trafo.iterrows():
            G.add_edge(int(tr.hv_bus), int(tr.lv_bus), element=int(tr.name))
        for _, tr in net.trafo3w.iterrows():
            G.add_edge(int(tr.hv_bus), int(tr.lv_bus))
            G.add_edge(int(tr.hv_bus), int(tr.mv_bus))
            G.add_edge(int(tr.lv_bus), int(tr.mv_bus))
    
    return G

def preprocess_network_for_optimization(net, logger=None, debug=False, graph_id=None, output_dir=None):
    if logger is None:
        logger = logging.getLogger(f"preprocess.{graph_id}")

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
        logger.warning(f"{graph_id}: No distribution components found.")
        return None, None, None
    
    main_dist_component = set(max(dist_components, key=len))
    logger.info(f"{graph_id}: Selected main distribution component with {len(main_dist_component)} buses.")
    
    # Create active bus mask for distribution buses only
    active_bus_mask = pd.Series(False, index=net.bus.index)
    active_bus_mask.loc[list(main_dist_component)] = True
    
    # Identify different types of excluded nodes
    G_full_phys = build_nx_graph(net, include_lines=True, include_trafos=True)
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
        logger.warning(f"{graph_id}: Selected main component has no original reference buses, adding one...")
        subnet_graph_for_slack = G_dist_only.subgraph(main_dist_component)
        degrees_in_main = dict(subnet_graph_for_slack.degree())
        if degrees_in_main:
            new_slack = max(degrees_in_main, key=degrees_in_main.get)
            slack_buses.add(new_slack)
            logger.warning(f"{graph_id}: Added new reference bus {new_slack} to slack_buses set.")

    # Create active bus mask for distribution buses only
    active_bus_mask = pd.Series(False, index=net.bus.index)
    active_bus_mask.loc[list(main_dist_component)] = True
    
    if debug:
        logger.debug(f"{graph_id}: Debugging graph connectivity:")
        logger.debug(f"{graph_id}: Selected main_dist_component size: {len(main_dist_component)}")
        logger.debug(f"{graph_id}: Main_dist_component buses: {sorted(list(main_dist_component))}")
        logger.debug(f"{graph_id}: Transformer-linked_nodes: {len(transformer_linked_nodes)}")
        logger.debug(f"{graph_id}: Other_excluded_nodes: {len(other_excluded_nodes)}")
        logger.debug(f"{graph_id}: All identified slack buses: {slack_buses}")
    
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
            logger.warning(f"{graph_id}: Processed network has no power sources, adding one...")
            candidate_buses_for_ext_grid = main_dist_component & slack_buses
            chosen_slack_bus_for_ext_grid = next(iter(candidate_buses_for_ext_grid), None)
            
            if not chosen_slack_bus_for_ext_grid: 
                subnet_graph = G_dist_only.subgraph(main_dist_component)
                degrees = dict(subnet_graph.degree())
                if degrees:
                    chosen_slack_bus_for_ext_grid = max(degrees, key=degrees.get)
            
            if chosen_slack_bus_for_ext_grid is not None and chosen_slack_bus_for_ext_grid in processed_net.bus.index:
                pp.create_ext_grid(processed_net, chosen_slack_bus_for_ext_grid, vm_pu=1.0, va_degree=0.0)
                logger.warning(f"{graph_id}: Added ext_grid at bus {chosen_slack_bus_for_ext_grid}.")
                slack_buses.add(chosen_slack_bus_for_ext_grid)

    except Exception as e:
        logger.error(f"{graph_id}: Error creating subnet: {e}")
        logger.error(traceback.format_exc())
        return None, None, None
    
    relevant_slack_buses = list(slack_buses & main_dist_component)
    
    visualization_data = {
        "main_component_nodes": list(main_dist_component),
        "transformer_linked_nodes": list(transformer_linked_nodes),
        "other_excluded_nodes": list(other_excluded_nodes),
        "slack_buses": relevant_slack_buses 
    }
    
    # if debug and graph_id is not None and output_dir is not None:
    #     try:
    #         visualize_preprocessing(net, active_bus_mask, graph_id, visualization_data, 
    #                                output_dir=output_dir, debug=debug)
    #     except Exception as e:
    #         logger.error(f"{graph_id}: Error during visualization: {e}")
    #         logger.error(traceback.format_exc())
    
    logger.info(f"{graph_id}: Preprocessed network: {len(processed_net.bus)} of {len(net.bus)} buses kept.")
    return processed_net, active_bus_mask, visualization_data

def visualize_preprocessing(net_orig, active_bus_mask, graph_id, visualization_data, output_dir=None, debug=False):
    if output_dir is None:
        output_dir = Path("data_generation") / "logs" / "visualizations"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    G = build_nx_graph(net_orig, include_lines=True, include_trafos=True)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.gca() 

    pos = {}
    if hasattr(net_orig, 'bus_geodata') and not net_orig.bus_geodata.empty:
        pos = {int(b): (r.x, r.y) for b, r in net_orig.bus_geodata.iterrows() if b in G}
    
    if not pos or len(pos) < G.number_of_nodes():
        if G.number_of_nodes() > 0:
            layout_pos = pos if pos else None
            fixed_nodes = list(pos.keys()) if pos else None
            current_pos = nx.spring_layout(G, seed=42, pos=layout_pos, fixed=fixed_nodes)
            pos.update(current_pos)
        else:
            pos = {}

    main_component_nodes = set(visualization_data.get("main_component_nodes", []))
    transformer_linked_nodes = set(visualization_data.get("transformer_linked_nodes", []))
    other_excluded_nodes = set(visualization_data.get("other_excluded_nodes", []))
    slack_buses_viz = set(visualization_data.get("slack_buses", []))

    active_edges = []
    inactive_edges = []
    if G.number_of_edges() > 0:
        for u, v in G.edges():
            if u in main_component_nodes and v in main_component_nodes:
                active_edges.append((u,v))
            else:
                inactive_edges.append((u,v))

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=inactive_edges, edge_color='lightcoral', width=1.0, alpha=0.6)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=active_edges, edge_color='royalblue', width=1.5, alpha=0.8)
    
    cat_main = [n for n in main_component_nodes if n in G]
    cat_trafo = [n for n in transformer_linked_nodes if n in G]
    cat_other = [n for n in other_excluded_nodes if n in G]
    cat_slack = [n for n in slack_buses_viz if n in G]

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=cat_main, 
                         node_color='blue', node_size=60, label=f'Main Component ({len(cat_main)})')
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=cat_trafo, 
                         node_color='pink', node_size=50, alpha=0.9, label=f'Excluded (Transformer-Linked) ({len(cat_trafo)})')
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=cat_other, 
                         node_color='orange', node_size=40, alpha=0.8, label=f'Excluded (Isolated/Other) ({len(cat_other)})')
    
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=cat_slack, 
                         node_color='green', node_size=250, 
                         node_shape='*', label=f'Slack/Reference Buses ({len(cat_slack)})')
    
    labels_to_draw = {bus: str(bus) for bus in cat_slack}
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels_to_draw, font_size=8, font_weight='bold')
    
    ax.legend(loc='upper right', scatterpoints=1, frameon=True, fontsize='small')
    ax.set_title(f"Preprocessing Results for {graph_id}", fontsize='medium')
    ax.axis('off')
    
    plt.tight_layout() 
    
    filename = output_dir / f"{graph_id}_preprocessing.png"
    fig.savefig(filename, bbox_inches='tight', dpi=300) 
    plt.close(fig) 

    print(f"Preprocessing visualization saved to {filename}") 

# def create_operational_graph(net):
#     G = nx.Graph()
#     G.add_nodes_from(net.bus.index)
    
#     # Add lines that have closed switches
#     line_switches = net.switch[net.switch.et == 'l']
    
#     for switch_idx, switch in line_switches.iterrows():
#         if switch.closed:  
#             line_idx = switch.element
#             line = net.line.loc[line_idx]
#             G.add_edge(int(line.from_bus), int(line.to_bus), 
#                       line_idx=int(line_idx), 
#                       switch_idx=int(switch_idx))
    
#     # Add bus-bus switches if closed
#     bus_switches = net.switch[net.switch.et == 'b']
#     for switch_idx, switch in bus_switches.iterrows():
#         if switch.closed:
#             G.add_edge(int(switch.bus), int(switch.element),
#                       switch_idx=int(switch_idx),
#                       is_bus_switch=True)
    
#     return G

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

def find_disconnected_buses(G):
    disconnected_buses = set()
    # Find isolated nodes and very small components
    for component in nx.connected_components(G):
        if len(component) <= 2:  # Isolated or very small components
            disconnected_buses.update(component)
        # Also check for nodes with degree 0 (completely isolated)
        for node in component:
            if G.degree(node) == 0:
                disconnected_buses.add(node)
    
    return disconnected_buses

def build_operational_graph(G):
    H = G.copy()
    H.remove_edges_from([e for e,d in G.edges(data=True) if d.get("closed",True) is False])
    return H

def find_cycles(G):
    cycle_edges = set()
    
    # Check each connected component for cycles
    for component in nx.connected_components(G):
        if len(component) <= 2:
            continue
            
        subgraph = G.subgraph(component)
        if not nx.is_tree(subgraph):
            # Find all cycles in this component
            try:
                cycles = nx.cycle_basis(subgraph)
                for cycle in cycles:
                    # Add all edges in the cycle
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        cycle_edges.add((min(u, v), max(u, v)))
            except:
                # Fallback: just mark that there are cycles
                for u, v in subgraph.edges():
                    cycle_edges.add((min(u, v), max(u, v)))
    
    return cycle_edges



def is_radial_and_connected(net, y_mask=None, include_lines=True, include_switches=True):

    G = build_nx_graph(net, include_lines=include_lines, include_switches=include_switches)
    # Apply mask by removing nodes if provided
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


def count_switch_changes(net_a, net_b):
    sa = net_a.switch[(net_a.switch.et=='l')]['closed']
    sb = net_b.switch[(net_b.switch.et=='l')]['closed']
    common = sa.index.intersection(sb.index)
    return int((sa.loc[common] != sb.loc[common]).sum())

def loss_improvement(net_before, net_after, include_trafos=True):
    pp.runpp(net_before, enforce_q_lims=False)
    pp.runpp(net_after,  enforce_q_lims=False)
    def active_loss(net):
        loss = net.res_line["pl_mw"].abs().sum()
        if include_trafos and "res_trafo" in net:
            loss += net.res_trafo["pl_mw"].abs().sum()
        return float(loss)

    l0 = active_loss(net_before)
    l1 = active_loss(net_after)

    return {
        "loss_before":       l0,
        "loss_after":        l1,
        "loss_improvement":  100.0 * (l0 - l1) / l0 if l0 > 0 else np.nan
    }



def plot_grid(net, ax, title=""):
    #G   = create_nxgraph(net, include_lines=True, include_switches=True)
    G = build_nx_graph(net,include_lines=True, include_switches=True)
    isolated = [n for n, d in G.degree() if d == 0]
    
    if not net.bus_geodata.empty:
        pos = {int(b): (r.x, r.y) for b, r in net.bus_geodata.iterrows()}
    else:
        pos = nx.kamada_kawai_layout(G)

    # Find disconnected buses and cycles
    disconnected_buses = find_disconnected_buses(G)
    cycle_edges = find_cycles(G)

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

    nx.draw_networkx_edges(G, pos, edgelist=cycle_edges_list, edge_color="black", ax=ax, width=1.5, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=open_edges, edge_color="red", ax=ax, width=2.0)
    
    
    # open switches
    open_edges = [(u, v) for u, v, d in G.edges(data=True)
              if not d.get("closed", True)]
    if open_edges:
        nx.draw_networkx_edges(G, pos, edgelist=open_edges,
                               edge_color="red", width=2.0, ax=ax)
        incident = {n for e in open_edges for n in e}
        nx.draw_networkx_nodes(G, pos, nodelist=list(incident),
                               node_size=75, node_color="red",
                               edgecolors="black", linewidths=1.5, ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=list(disconnected_buses), node_size=80,
                          node_color="black", edgecolors="black", linewidths=2, ax=ax)
    
    ax.set_title(title+ "open switches: " + str(len(open_edges)))
    ax.axis("off")


def plot_voltage_profile(voltage_data, ax, bus_labels=None, title="Voltage profile"):
    voltage_data = np.asarray(voltage_data, dtype=float)
    
    num_phases = 0
    if voltage_data.ndim > 1:
        num_phases = voltage_data.shape[1]
    elif voltage_data.ndim == 1 and voltage_data.size > 0 :
        num_phases = 1 
        voltage_data = voltage_data[:, np.newaxis] 
    else: # Empty data
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

# def create_nxgraph(net, *, include_lines=True, include_switches=True):
#     G = nx.Graph()
#     # plain lines -----------------------------------------------------------
#     if include_lines:
#         for idx, ln in net.line.iterrows():
#             G.add_edge(int(ln.from_bus), int(ln.to_bus),
#                        element=int(idx),
#                        closed=True)                   
#     # line-switches ---------------------------------------------------------
#     if include_switches:
#         for s, sw in net.switch.query("et == 'l'").iterrows():
#             ln   = net.line.loc[int(sw.element)]
#             edge = (int(ln.from_bus), int(ln.to_bus))
#             G.edges[edge]["closed"] = bool(sw.closed)
#             G.edges[edge]["switch_id"] = int(s)      
#     return G



def visualize_network_states(snapshots, active_bus_mask, graph_id, output_dir=None, debug=False):
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
        return
    
    fig, axes = plt.subplots(1, n_nets + 1, figsize=(4 * (n_nets + 1), 5))
    if n_nets == 0: 
        axes = [axes]

    common_ref_bus = None
    for net in nets_to_visualize:
        if hasattr(net, 'ext_grid') and not net.ext_grid.empty:
            common_ref_bus = net.ext_grid.bus.iloc[0]
            break
    
    # Get union of all bus indices to ensure consistent indexing
    all_bus_indices = set()
    for net in nets_to_visualize:
        if hasattr(net, "bus"):
            all_bus_indices.update(net.bus.index.tolist())
    reference_bus_index = pd.Index(sorted(list(all_bus_indices)))
    logger.debug(f"Reference bus index: {reference_bus_index}")
   

    aligned_voltage_data = []
    for i, net in enumerate(nets_to_visualize):
        try:
            # Ensure power flow is solved with same reference
            if common_ref_bus is not None and common_ref_bus in net.bus.index:
                # Set consistent reference voltage
                if hasattr(net, 'ext_grid') and not net.ext_grid.empty:
                    net.ext_grid.loc[:, 'vm_pu'] = 1.0
                    net.ext_grid.loc[:, 'va_degree'] = 0.0
            
            if not hasattr(net, 'res_bus') or net.res_bus.empty:
                pp.runpp(net, enforce_q_lims=False, calculate_voltage_angles=False)
        except Exception as e:
            logger.warning(f"Power flow failed for {labels[i]}: {e}")

        if hasattr(net, 'res_bus') and not net.res_bus.empty and 'vm_pu' in net.res_bus.columns:
            current_voltages = net.res_bus.vm_pu.reindex(reference_bus_index, fill_value=np.nan)
            aligned_voltage_data.append(current_voltages.values)
        else:
            aligned_voltage_data.append(np.full(len(reference_bus_index), np.nan))

    
    for i, (net, label) in enumerate(zip(nets_to_visualize, labels)):
        current_ax = axes[i] if n_nets > 0 else axes 
        plot_grid(net, current_ax, title=label)
        
        mask_for_check = None
        if label.lower() == "original":
            mask_for_check = None
        elif hasattr(net, "bus") and not net.bus.empty and active_bus_mask is not None:
            if isinstance(active_bus_mask, pd.Series):
                 mask_for_check = active_bus_mask.reindex(net.bus.index, fill_value=False)
            else: 
                 mask_for_check = active_bus_mask

        # Get detailed topology status
        #status = get_network_topology_status(net, y_mask=mask_for_check)
        
        # Create status text
        op_status = "r"# f"Op: R={status['operational']['radial']} C={status['operational']['connected']}"
        topo_status = "r"#f"Topo: R={status['topological']['radial']} C={status['topological']['connected']}"
        comp_info ="r"# f"Comp: {status['operational_components']}/{status['topological_components']}"
        
        current_ax.text(0.5, -0.12,
                    f"{op_status}\n{topo_status}\n{comp_info}",
                    transform=current_ax.transAxes,
                    ha="center", va="top", fontsize=7)
    
    voltage_plot_ax = axes[n_nets] if n_nets > 0 else axes 
    if aligned_voltage_data:
        stacked_voltages = np.column_stack(aligned_voltage_data)
        bus_labels_for_plot = [str(i) for i in reference_bus_index.tolist()]
        plot_voltage_profile(stacked_voltages, voltage_plot_ax, bus_labels=bus_labels_for_plot, title="Voltage Profile")
    else:
        voltage_plot_ax.text(0.5, 0.5, "Voltage profile data not available.", ha='center', va='center')
        voltage_plot_ax.set_title("Voltage Profile")
        voltage_plot_ax.axis('off')

    plt.tight_layout()
    
    if output_dir:
        filename = output_dir / f"{graph_id}_network_states.png"
        try:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        except Exception as e:
            pass
        plt.close(fig)
    else:
        plt.show()


def store_snapshots(graph_id: str, root_folder: Path, logger, **nets,):
    for phase, net in nets.items():
        out_dir = root_folder / phase / "pandapower_networks"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{graph_id}.json"         
        with open(fname, "w") as fp:
            fp.write(pp.to_json(net))

        logger.info(f"[{graph_id}] snapshot '{phase}' saved → {fname}")


def process_single_graph(graph_id, net_json, folder_path, toggles, vis_dir=None, lp_dir=None, logger=None, debug=False):
    logger.debug(f"Processing graph {graph_id}")
    logger = get_logger(f"graph.{graph_id}")
    net_orig = pp.from_json_string(net_json)
    try: 
        pp.runpp(net_orig, enforce_q_lims=False)
        if not net_orig.converged:
            logger.warning(f"{graph_id}: PF on original net failed - skip")
            return None
    except LoadflowNotConverged:
        logger.warning(f"{graph_id}: PF on original net failed - skip")
        return None
    
    pdb.set_trace()
    # Preprocess network to ensure connectivity
    processed_net, active_bus_mask, viz_data = preprocess_network_for_optimization(net_orig, debug=debug, graph_id=graph_id, output_dir=vis_dir,logger =logger)
    
    if processed_net is None:
        logger.warning(f"{graph_id}: Network preprocessing failed - cannot create connected component")
        return None
    
    try:
        pp.runpp(processed_net, enforce_q_lims=False)
    except LoadflowNotConverged:
        logger.warning(f"{graph_id}: PF on processed network failed - skip")
        return None
    pdb.set_trace()
    # Apply MST reconfiguration
    net_mst = alternative_mst_reconfigure(copy.deepcopy(processed_net), penalty=1.0, y_mask=active_bus_mask)
    
    pdb.set_trace()
    # Check radiality and connectivity after MST
    rad_conn_mst = is_radial_and_connected(net_mst, y_mask=active_bus_mask)
    logger.info(f"{graph_id}: MST network is radial={rad_conn_mst[0]}, connected={rad_conn_mst[1]}")
    logger.debug(f"   >> net_mst.bus.count = {len(net_mst.bus.index)}")
    logger.debug(f"   >> active_bus_mask.count = {len(active_bus_mask)}")
    logger.debug(f"   >> active_bus_mask.index  = {list(active_bus_mask.index[:10])}...")  # first few
    try:
        pp.runpp(net_mst, enforce_q_lims=False)
    except LoadflowNotConverged:
        logger.warning(f"{graph_id}: PF after MST did not converge – skip")
        return None
    
    def align_mask_to_net(mask: pd.Series, net: pp.pandapowerNet) -> pd.Series:
        # forces mask to exactly the net.bus.index
        return mask.reindex(net.bus.index, fill_value=False).astype(bool)

    # … then in process_single_graph:
    mask_mst = align_mask_to_net(active_bus_mask, net_mst)

    # Apply SOCP optimization
    net_opt_mst, metrics_mst = apply_socp(
        net_mst, 
        graph_id + "_mst", 
        toggles=toggles, 
        logger=logger,
        active_bus_mask=active_bus_mask,
        lp_dir=lp_dir,
        debug=debug
    )
    
    flips_orig_to_mst = count_switch_changes(processed_net, net_mst)
    flips_mst_to_opt = count_switch_changes(net_mst, net_opt_mst)
    
    metrics = {
        "graph_id": graph_id,
        "total_switches": processed_net.switch.shape[0],
        "switches_changed_orig_to_mst": flips_orig_to_mst,
        "switches_changed_mst_to_opt": flips_mst_to_opt,
        "opt_time_mst": metrics_mst["opt_time"],
        "loss_improvement_mst": metrics_mst["loss_improvement"],
        "rad_mst": rad_conn_mst,
        "rad_mst_opt": (metrics_mst["after_radial"], metrics_mst["after_connected"]),
    }
    
    snapshots = {
        "original": net_orig,
        "processed": processed_net,
        "mst": net_mst,
        "optimised_mst": net_opt_mst,
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
            snapshots["optimised_orig"] = net_opt_orig
            logger.info(
                f"{graph_id}: MST→OPT={flips_mst_to_opt} flips; origPF→OPT={metrics_orig['switches_changed']} flips; diff={flips_mst_vs_orig}"
            )
        except Exception as e:
            logger.warning(f"{graph_id}: Error in original network optimization: {e}")
            logger.debug(traceback.format_exc())
    else:
        logger.info(f"{graph_id}: MST only, MST→OPT={flips_mst_to_opt} flips")
    
    visualize_network_states(snapshots, active_bus_mask, graph_id, output_dir=vis_dir, debug=debug)
    store_snapshots(graph_id, folder_path, logger, **snapshots)
    
    return metrics


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
    
    # Run verification
    if debug:
        logger.info("Running detailed constraint verification...")
        optimizer.verify_power_balance_constraints()
        optimizer.verify_voltage_drop_constraints()
        optimizer.verify_socp_cone_constraints()
        optimizer.verify_line_flow_bounds()
    # Update the network with optimization results
    net_opt = optimizer.update_network()
    
    # Try to run power flow on the optimized network
    pf_converged = False
    try:
        pp.runpp(net_opt, enforce_q_lims=False)
        pf_converged = net_opt.converged
    except LoadflowNotConverged:
        logger.warning(f"{graph_id}: PF after optimization did not converge")
    
    # Calculate metrics
    flips = count_switch_changes(net_before_opt, net_opt)
    rad_before = is_radial_and_connected(net_before_opt, y_mask=active_bus_mask)
    rad_after = is_radial_and_connected(net_opt, y_mask=active_bus_mask)
    loss_impr = None
    if pf_converged:
        loss_impr = loss_improvement(net_before_opt, net_opt)

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

def apply_optimization(folder_path, method="SOCP", toggles=None, debug=False, serialize=False):
    folder_path = Path(folder_path)
    root_logger = init_application_logging(method=method, debug=debug)
    logger = get_logger("optimization")
    logger.info(f"Starting {method} optimization on {folder_path}")
    logger.info(f"Toggles: {toggles}")

    if not serialize:
        init_worker_logging()


    _, pp_networks, features = load_graph_data_old(folder_path)
    items = [(gid, net) for gid, net in pp_networks.items()]
    metrics = []
    vis_dir = Path("data_generation")  / "logs" / "visualizations"
    lp_dir = Path("data_generation") / "logs" / "lp_files"
    
    vis_dir.mkdir(parents=True, exist_ok=True)
    lp_dir.mkdir(parents=True, exist_ok=True)

    if serialize:
            #--- Sequential execution ---
        for gid, net_json in items:
            res = process_single_graph(gid, net_json, folder_path , toggles,vis_dir,lp_dir,logger=logger, debug = debug)
            if res: metrics.append(res)
    else: 
        # --- Parallel execution ---
        with ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            initializer=init_worker_logging
        ) as executor:
            futures = {
                executor.submit(
                    process_single_graph,
                    gid, net_json, folder_path,
                    toggles=toggles, debug=debug
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

        plt.tight_layout()
        plt.show()
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ground truth data for power networks using optimization')
    parser.add_argument('--folder_path',
                        default = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32/test/original",
                        #default = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\_synthetic-train-data_12052025_range-130-100_3\original",
                        #default= r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_val_real__range-30-230_nTest-10_nVal-10_1252025_14\test",
                        #default = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_val_real__range-30-230_nTest-10_nVal-10_1252025_2\test",
                        type=str, help='Dataset folder path')
    parser.add_argument('--set', type=str, choices=['test', 'validation', 'train', '', 'all'], default='', help='Dataset set to process; leave empty for no subfolder')
    parser.add_argument('--method', type=str, choices=['SOCP', 'MILP'], default='SOCP', help='Choose optimization method: SOCP or MILP')
    parser.add_argument('--debug', type=bool, default=True, help='Print debug information')
    parser.add_argument('--serialize', type=bool, default=True, help='Serialize the optimization results usefull for debugging')
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

    logger = init_application_logging(method=args.method, debug=args.debug)
    
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
    if args.set:
        apply_optimization(Path(args.folder_path) / args.set, method=args.method, toggles=SOCP_toggles, debug=args.debug, serialize=args.serialize)
    elif args.set == "all": 
        for set_name in Path(args.folder_path).iterdir():
            if set_name.is_dir():
                print("\nProcessing set:", set_name)
                apply_optimization(Path(args.folder_path) / set_name, method=args.method, toggles=SOCP_toggles, debug=args.debug, serialize=args.serialize)
    else:
        apply_optimization(args.folder_path, method=args.method, toggles=SOCP_toggles, debug=args.debug, serialize=args.serialize)

    print("\nGround truth generation complete!!!!")

