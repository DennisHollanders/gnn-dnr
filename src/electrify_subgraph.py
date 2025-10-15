import os
import copy
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import partial
from functools import lru_cache
import glob
import geopandas as gpd
import itertools   
import json 
import logging
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import networkx as nx
import pandas as pd
import pickle as pkl
import pandapower as pp
import pandapower.networks as pn 
import pandapower.topology as top
from pathlib import Path
import random 
import re
from scipy.spatial import ConvexHull
from shapely.geometry import Point, LineString
import string
import simbench as sb
from typing import List, Dict, Tuple, Any
import time
from tqdm import tqdm

import sys 

random.seed(0)
logger = logging.getLogger("data_generation")

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


STANDARD_CABLE_MAPPING = {
        'standard_cable_1': {
            "r_ohm_per_km": 0.124, "x_ohm_per_km": 0.080,  
            "c_nf_per_km": 280, "max_i_ka": 0.32,
            "q_mm2": 150, "alpha": 0.00403, "type": "cs"
        },
        'standard_cable_2': {
            "r_ohm_per_km": 0.162, "x_ohm_per_km": 0.070,  
            "c_nf_per_km": 310, "max_i_ka": 0.28,
            "q_mm2": 240, "alpha": 0.00403, "type": "cs"
        },
        'standard_cable_3': {
            "r_ohm_per_km": 0.193, "x_ohm_per_km": 0.080,  
            "c_nf_per_km": 210, "max_i_ka": 0.25,
            "q_mm2": 95, "alpha": 0.00403, "type": "cs"
        }
    }

def sample_from_distribution_global(dist_name: str, distribution: Dict[str, Any]) -> Any:
    dist_type = distribution["type"]
    if dist_type == "normal":
        sample = np.random.normal(distribution["mean"], distribution["std"])
        sample = np.clip(sample, distribution["min"], distribution["max"])
        if distribution.get("is_integer", False):
            sample = int(round(sample))

    elif dist_type == "uniform":
        sample = np.random.uniform(distribution["min"], distribution["max"])
        if distribution.get("is_integer", False):
            sample = int(round(sample))

    elif dist_type == "beta":
        raw_sample = np.random.beta(distribution["alpha"], distribution["beta"])
        sample = distribution["min"] + raw_sample * (distribution["max"] - distribution["min"])
        if distribution.get("is_integer", False):
            sample = int(round(sample))

    elif dist_type == "discrete":
        sample = np.random.choice(distribution["values"])

    elif dist_type == "categorical":
        choices = distribution["choices"]
        weights = distribution.get("weights", [1 / len(choices)] * len(choices))
        sample = random.choices(choices, weights=weights, k=1)[0]

    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")
    logger.debug(f"Sampled using distcallable: {dist_name} -> {sample}")
    return sample

def initialize_distributions(distributions: Dict[str, dict]) -> Dict[str, Any]:
    """
    Returns a dict of dist_name -> callable() for sampling.
    """
    return {
        name: partial(sample_from_distribution_global, name, params)
        for name, params in distributions.items()
    }

def retrieve_standard_production(timestamp, df):
    """Enhanced version with better scaling"""
    timestamp = pd.to_datetime(timestamp)
    A_columns = [col for col in df.columns if col.endswith("_A")]
    A_col = random.choice(A_columns)
    I_col = A_col.rsplit("_A",1)[0] +"_I"
    row = df[pd.to_datetime(df["from"]) == timestamp]
    
    consumption = row[A_col].values[0] if len(row) > 0 else 0.1
    production = row[I_col].values[0] if len(row) > 0 else 0.05
    
    all_consumption = df[A_col].values
    all_production = df[I_col].values
    
    p75_consumption = np.percentile(all_consumption[all_consumption > 0], 75)
    p75_production = np.percentile(all_production[all_production > 0], 75)
    
    return consumption, production, p75_consumption, p75_production

def sample_timeframes(num_intervals, interval_duration_minutes=15):
    start_of_year = pd.Timestamp("2025-01-01 00:00:00")
    end_of_year = pd.Timestamp("2025-12-31 23:59:59")
    total_minutes = (end_of_year - start_of_year).total_seconds() / 60
    max_intervals = int(total_minutes / interval_duration_minutes)
    random_start_interval = random.randint(0, max_intervals - num_intervals)
    random_start_time = start_of_year + timedelta(minutes=random_start_interval * interval_duration_minutes)
    timeframes = [ random_start_time + timedelta(minutes=i * interval_duration_minutes) for i in range(num_intervals)]
    
    return timeframes
def runpp_with_auto_scale(net, args, max_iterations=5, scale_floor=0.5):
    scale_factor = 1
    for it in range(max_iterations):
        logger.info("scalin down loads by factor: {scale_factor:.2f}")
        pp.runpp(net, algorithm='nr', max_iteration=50)
        if not net.converged:
            scale_factor *= 0.75
            if scale_factor < scale_floor:
                return False, net, scale_factor
            net.load["p_mw"] *= 0.75
            net.sgen["p_mw"] *= 0.75
            net.gen.loc[net.gen.slack == False, "p_mw"] *= 0.8
            continue

        # Check voltages and line loadings
        vm_min = net.res_bus.vm_pu.min()
        max_line_load = net.res_line.loading_percent.max()

        if vm_min < 0.90 or max_line_load > args.max_line_loading:
     
            scale_factor *= 0.9
            if scale_factor < scale_floor:
                return False, net, scale_factor
            net.load["p_mw"] *= 0.9
            net.sgen["p_mw"] *= 0.9
            net.gen.loc[net.gen.slack == False, "p_mw"] *= 0.9
        else:
            return True, net, scale_factor

    return False, net, scale_factor

def assign_realistic_loads(modified_subgraph, args, dist_callables, timeframes, dfs):

    date_time = timeframes[0]
    consumption, production, p75_cons, p75_prod = retrieve_standard_production(date_time, dfs[3])
    hour = pd.to_datetime(date_time).hour
    if 6 <= hour < 10:
        time_factor = 0.7  
    elif 17 <= hour < 21:
        time_factor = 1.0  
    elif 22 <= hour < 6:
        time_factor = 0.3  
    else:
        time_factor = 0.5  
    
 
    global_scale = (consumption / p75_cons if p75_cons > 0 else 0.5) * time_factor
    global_scale = np.clip(global_scale, 0.3, 0.8)
   
    node_cable_capacities = analyze_node_cable_capacity(modified_subgraph)
 
    total_capacity = sum(node_cable_capacities.values())
    avg_capacity = np.mean(list(node_cable_capacities.values()))

    target_loading_min = 0.2 * args.max_line_loading  
    target_loading_max = 0.6 * args.max_line_loading 
    
    # Assign loads based on cable capacity
    node_loads = {}
    total_load_preliminary = 0
    for node in modified_subgraph.nodes:
        node_capacity = node_cable_capacities.get(node, avg_capacity)
 
        voltage_kv = 10.0  
        
        # P = sqrt(3) * V * I * cos(phi)
        # For balanced three-phase: P = 1.732 * V * I * pf
        power_factor = random.uniform(0.85, 0.90)  #
        
        # safety factor 0.6
        max_node_power = 1.732 * voltage_kv * node_capacity * power_factor* 0.6  # MW
    
        loading_fraction = random.uniform(target_loading_min / 100, target_loading_max / 100)
        base_load = max_node_power * loading_fraction

        load_variation = random.uniform(0.8, 1.2)
        raw_load = base_load * load_variation * global_scale
        consumption_value = float(np.minimum(raw_load, max_node_power))

        consumption_value = max(consumption_value, 1e-4)

        node_degree = modified_subgraph.degree(node)
        dg_global = dist_callables["dg_penetration"]()
        weight = 0.5 + 0.5 * (node_degree / max(1, max(dict(modified_subgraph.degree()).values())))
        production_value = 0.0
        logger.info("dg_value: {dg_global}, weight: {weight}, node_degree: {node_degree}")
        if random.random() < dg_global * weight:
            dg_factor = random.uniform(0.1, 0.3) 
            production_value = consumption_value * dg_factor * random.uniform(0.3, 1.0)
  
        net_load = consumption_value - production_value
        
        # Store the values
        node_loads[node] = {
            'net_load': net_load,
            'consumption': consumption_value,
            'production': production_value,
            'power_factor': power_factor,
            'cable_capacity': node_capacity
        }
        
        total_load_preliminary += consumption_value - production_value
    
    # Ensure total network loading is reasonable
    total_cable_capacity_mw = 1.732 * 10.0 * total_capacity * 0.9 * 0.6  
    max_network_load = total_cable_capacity_mw * 0.4  
    
    if total_load_preliminary > max_network_load:
        scale_factor = max_network_load / total_load_preliminary
        logger.info(f"Scaling down loads by factor {scale_factor:.2f} to ensure convergence")
        
        for node in node_loads:
            node_loads[node]['consumption'] *= scale_factor
            node_loads[node]['production'] *= scale_factor
            node_loads[node]['net_load'] = (node_loads[node]['consumption'] - 
                                           node_loads[node]['production'])
    
    # Apply simplified power flow constraints
    adjusted_loads = apply_power_flow_constraints(
        modified_subgraph, node_loads, args.max_line_loading
    )
    
    # Assign the adjusted loads to the graph
    for node, load_data in adjusted_loads.items():
        modified_subgraph.nodes[node]["net_load"] = load_data['net_load']
        modified_subgraph.nodes[node]["power_factor"] = load_data['power_factor']
        modified_subgraph.nodes[node]["cable_capacity"] = load_data['cable_capacity']
    
    # ensure at least some load exists
    total_final_load = sum(data['net_load'] for data in adjusted_loads.values() if data['net_load'] > 0)
    logger.info(f"Final total load: {total_final_load:.3f} MW")
    
    return modified_subgraph

def analyze_node_cable_capacity(subgraph):
    node_capacities = {}
    
    for node in subgraph.nodes:
        total_capacity = 0
        connected_capacities = []
        
        for neighbor in subgraph.neighbors(node):
            edge_data = subgraph.edges[node, neighbor]
      
            cable_props = edge_data.get("pandapower_type", {})
            if isinstance(cable_props, dict):
                max_i_ka = cable_props.get("max_i_ka", 0.3)  
            else:
                max_i_ka = 0.3
            
            connected_capacities.append(max_i_ka)   
        if connected_capacities:
            total_capacity = sum(connected_capacities)
            avg_capacity = np.mean(connected_capacities)
            node_capacities[node] = 0.7 * avg_capacity + 0.3 * (total_capacity / len(connected_capacities))
        else:
            node_capacities[node] = 0.1
    
    return node_capacities

def add_reactive_compensation(net):
    if len(net.load) > 0 and len(net.res_bus) > 0:
        high_load_buses = net.load.groupby('bus')['p_mw'].sum().nlargest(5).index    
        for bus in high_load_buses:
            if bus in net.res_bus.index:
                bus_load = net.load[net.load.bus == bus]['q_mvar'].sum()
                if bus_load > 0.5: 
                    cap_size = bus_load * 0.3
                    pp.create_shunt(net, bus=bus, q_mvar=-cap_size, p_mw=0)
                    logger.debug(f"Added {cap_size:.2f} MVar capacitor at bus {bus}")


def apply_power_flow_constraints(subgraph, node_loads, max_loading_percent):
    conservative_max_loading = max_loading_percent * 0.7 
    edge_loadings = estimate_edge_loadings(subgraph, node_loads)

    problem_edges = []
    for edge, loading in edge_loadings.items():
        if loading > conservative_max_loading:
            problem_edges.append((edge, loading))
    
    if not problem_edges:
        return node_loads
    
    logger.info(f"Found {len(problem_edges)} edges that might cause convergence issues")
    adjusted_loads = copy.deepcopy(node_loads)
    for (u, v), loading in problem_edges:
        reduction_factor = (conservative_max_loading / loading) * 0.8  
        
        for node in [u, v]:
            if node in adjusted_loads and adjusted_loads[node]['net_load'] > 0:
                adjusted_loads[node]['net_load'] *= reduction_factor
                adjusted_loads[node]['consumption'] *= reduction_factor
                logger.debug(f"Reduced load at node {node} by factor {reduction_factor:.2f}")
    return adjusted_loads


def estimate_edge_loadings(subgraph, node_loads):
    edge_loadings = {}
    for u, v, edge_data in subgraph.edges(data=True):

        cable_props = edge_data.get("pandapower_type", {})
        if isinstance(cable_props, dict):
            max_i_ka = cable_props.get("max_i_ka", 0.3)
        else:
            max_i_ka = 0.3
        u_load = node_loads[u]['net_load'] if u in node_loads else 0
        v_load = node_loads[v]['net_load'] if v in node_loads else 0

        load_diff = abs(u_load - v_load)
        
        # Estimate current (very simplified)
        # P = sqrt(3) * V * I * pf -> I = P / (sqrt(3) * V * pf)
        voltage_kv = 10.0  
        avg_pf = 0.9
        estimated_current = load_diff / (1.732 * voltage_kv * avg_pf)
        loading_percent = (estimated_current / max_i_ka) * 100
        edge_loadings[(u, v)] = loading_percent
    
    return edge_loadings


def find_switch_edges_in_non_radial(subgraph_to_adapt):
    for u, v, data in subgraph_to_adapt.edges(data=True):
        data['weight'] = data.get('length', 1.0)  
    mst = nx.minimum_spanning_tree(subgraph_to_adapt, weight='weight')
    mst_edges = set(mst.edges)
    all_edges = list(subgraph_to_adapt.edges(data=False))

    switch_edges = [edge for edge in all_edges if edge not in mst_edges]
    for u, v in mst_edges:
        subgraph_to_adapt.edges[u, v]['is_switch'] = False
        subgraph_to_adapt.edges[u, v]['is_synthetic'] = False
    for u, v in switch_edges:
        subgraph_to_adapt.edges[u, v]['is_switch'] = True
        subgraph_to_adapt.edges[u, v]['is_synthetic'] = False
    return subgraph_to_adapt, len(switch_edges)


def load_subgraphs(args) -> List[Any]:

    folder_path = args.subgraph_folder
    logger.info(f"Loading from folder: {folder_path}")
    file_names = os.listdir(folder_path)
    subgraphs = []

    if args.iterate_all:
        logger.info("Loading all .pkl files")
        for file_name in tqdm(file_names, desc="Loading files", unit="file"):
            full_path = os.path.join(folder_path, file_name)
            try:
                with open(full_path, 'rb') as f:
                    data = pkl.load(f)
                # extend if list or dict
                if isinstance(data, list):
                    subgraphs.extend(data)
                elif isinstance(data, dict) and 'subgraphs' in data:
                    subgraphs.extend(data['subgraphs'])
            except Exception as e:
                logger.error(f"Failed to load {full_path}: {e}")
    else:
        bus_range_max = args.target_busses + args.bus_range
        bus_range_min = args.target_busses - args.bus_range
        logger.info(f"Filtering by node range {bus_range_min}-{bus_range_max}")
        for file_name in tqdm(file_names, desc= "Loading files", unit="file"):
            nums = re.findall(r"(\d+)", file_name)
            if not nums:
                continue
            token = nums[0]
            half = len(token) // 2
            lower = int(token[:half])
            upper = int(token[half:])
            logger.info(f"File {file_name}: range {lower}-{upper}")
            # check overlap
            if lower <= bus_range_max and upper >= bus_range_min:
                full_path = os.path.join(folder_path, file_name)
                try:
                    with open(full_path, 'rb') as f:
                        data = pkl.load(f)
                    if isinstance(data, list):
                        subgraphs.extend(data)
                    elif isinstance(data, dict) and 'subgraphs' in data:
                        subgraphs.extend(data['subgraphs'])
                except Exception as e:
                    logger.error(f"Failed to load {full_path}: {e}")
    return subgraphs

def modify_subgraph(subgraph, args, dist_callable):
    if nx.is_frozen(subgraph):
        subgraph = nx.Graph(subgraph)     
    logger.info('Create new modified graph \n ___________________________')
    order_added = 0
    failed_attempts = 0 

    # Initially mark existing edges (which form the MST) as non‐switch
    subgraph_to_adapt, n_cycles = find_switch_edges_in_non_radial(subgraph)
    max_distance = calculate_max_distance(subgraph_to_adapt)
    max_cycle_length = calculate_max_theoretical_cycle_length(subgraph_to_adapt)
    logger.info(f"initial cycles: {n_cycles}, max cycle length: {max_cycle_length}, max distance: {max_distance}")
    required_switches = dist_callable['n_switches']()
    convex_layers_to_use = dist_callable['layer_list']()
    layers = extract_layers(nx.get_node_attributes(subgraph_to_adapt, 'position'), max(convex_layers_to_use))

    while n_cycles < required_switches:
        subgraph_to_adapt, succes = add_edge_via_convex_layers(
            subgraph_to_adapt, layers, convex_layers_to_use, max_distance, max_cycle_length, order_added, args
        )
        
        if succes:
            logger.debug("Added an edge")
            n_cycles += 1
            order_added += 1
        else:
            convex_layers_to_use = list(set([layer + 1 for layer in convex_layers_to_use]) | set([layer - 1 for layer in convex_layers_to_use]) | {0})
            failed_attempts += 1
            logger.debug(f"failed attempt to add an edge updated convex layers to use:{convex_layers_to_use}")
            if failed_attempts > 10:
                logger.info("Failed to add edges. Stopping.")
                break

    logger.info(f"cycles_added: {order_added}")
    logger.info(f"convex_layers_used: {convex_layers_to_use}")
    if args.plot_added_edge:
        fig = plt.figure(figsize=(12, 6))
        pos = nx.get_node_attributes(subgraph_to_adapt, 'position')
        nx.draw_networkx_nodes(subgraph_to_adapt, pos, node_color='black', node_size=10)
        nx.draw_networkx_edges(subgraph_to_adapt, pos, edge_color='black')
        synthetic_edges = [(u, v) for u, v, d in subgraph_to_adapt.edges(data=True) if d.get('is_synthetic', False)]
        if synthetic_edges:
            nx.draw_networkx_edges(subgraph_to_adapt, pos, edgelist=synthetic_edges, edge_color='red', width=2)
            synthetic_nodes = set()
            for u, v in synthetic_edges:
                synthetic_nodes.add(u)
                synthetic_nodes.add(v)
            nx.draw_networkx_nodes(subgraph_to_adapt, pos, nodelist=list(synthetic_nodes), node_color='red', node_size=20)
        plt.savefig(f'plot_of_added_edge_{subgraph}.png')
        plt.show()
    
    return subgraph_to_adapt

def calculate_max_theoretical_cycle_length(subgraph):
    """Calculates the maximum theoretical cycle length by connecting two longest paths."""
    if nx.is_empty(subgraph):
        return 0

    longest_paths = []
    for node in subgraph.nodes():
        path_lengths = nx.shortest_path_length(subgraph, source=node)
        longest_paths.append(max(path_lengths.values(), default=0))

    longest_paths = sorted(longest_paths, reverse=True)

    if len(longest_paths) >= 2:
        return longest_paths[0] + longest_paths[1] + 1 
    elif len(longest_paths) == 1:
        return longest_paths[0] + 1  
    return 0

def calculate_max_distance(subgraph_to_adapt):
    """Calculates the maximum edge distance for normalization."""
    pos = nx.get_node_attributes(subgraph_to_adapt, 'position')
    return max( np.linalg.norm(np.array(pos[u]) - np.array(pos[v])) for u, v, in subgraph_to_adapt.edges())

def extract_layers(pos,max_layer):
    """Extracts convex layers from node positions."""
    nodes = np.array(list(pos.values()))
    node_keys = list(pos.keys())
    remaining_nodes = nodes.copy()
    remaining_keys = node_keys.copy()
    layers = []
    while len(layers) <max_layer or len(remaining_nodes) >=3:
        hull = ConvexHull(remaining_nodes)
        layer_keys = [remaining_keys[i] for i in hull.vertices]
        layers.append(layer_keys)
        remaining_nodes = np.delete(remaining_nodes, hull.vertices, axis=0)
        remaining_keys = [remaining_keys[i] for i in range(len(remaining_keys)) if i not in hull.vertices]
    return layers


def add_edge_via_convex_layers(subgraph_to_adapt, layers, layer_list, max_distance, max_cycle_length, order_added, args):
    pos = nx.get_node_attributes(subgraph_to_adapt, 'position')
    selected_nodes = set(node for layer_index in layer_list if layer_index < len(layers) for node in layers[layer_index])
    selected_nodes = list(selected_nodes)

    possible_edges = []

    if args.within_layers:

        for i in range(len(selected_nodes)):
            for j in range(i + 1, len(selected_nodes)):
                u, v = selected_nodes[i], selected_nodes[j]
                if subgraph_to_adapt.has_edge(u, v):
                    continue
                edge = (u, v)
                score = check_and_add_edge(edge, subgraph_to_adapt, pos,
                                        max_distance, max_cycle_length,
                                        args.weight_factor)
                if score is not None:
                    possible_edges.append((edge, score))
    else:
        # loop over each layer, only adjacent pairs in that layer
        for layer in layers:
            for i in range(len(layer)):
                j = (i + 1) % len(layer)
                u, v = layer[i], layer[j]
                if subgraph_to_adapt.has_edge(u, v):
                    continue
                edge = (u, v)
                score = check_and_add_edge(edge, subgraph_to_adapt, pos,
                                        max_distance, max_cycle_length,
                                        args.weight_factor)
                if score is not None:
                    possible_edges.append((edge, score))
    possible_edges.sort(key=lambda x: x[1], reverse=True)

    if not possible_edges:
        logger.info("No valid edges to add.")
        return subgraph_to_adapt, False

    if args.deterministic:
        best_edge = possible_edges[0][0]
    else:
        top_edges = possible_edges[:args.top_x]
        scores = [score for _, score in top_edges]
        weights = np.exp(scores) / np.sum(np.exp(scores))
        selected_index = random.choices(range(len(top_edges)), weights=weights, k=1)[0]
        best_edge = top_edges[selected_index][0]

    subgraph_to_adapt.add_edge(
        best_edge[0], best_edge[1],
        is_switch=True,
        is_synthetic=True,
        order_added=order_added,
        geodata=(pos[best_edge[0]], pos[best_edge[1]])
    )

    return subgraph_to_adapt, True

def check_and_add_edge(edge, subgraph, pos, max_distance, max_cycle_length, weight_factor):
    """Checks an edge and calculates its score if it does not intersect existing edges."""
    new_edge = LineString([pos[edge[0]], pos[edge[1]]])
    intersects = any(new_edge.crosses(LineString([pos[u], pos[v]])) for u, v in subgraph.edges())

    if not intersects: 
        if not subgraph.has_edge(*edge):
            temp_graph = subgraph.copy()
            temp_graph.add_edge(*edge)
            try:
                cycles = nx.cycle_basis(temp_graph)
                cycle_lengths = [len(cycle) for cycle in cycles if edge[0] in cycle and edge[1] in cycle]
                if cycles and cycle_lengths:
                    cycle_length = min(cycle_lengths)

                    if max_cycle_length > 0:
                        normalized_cycle_length = cycle_length / max_cycle_length
                    else:
                        normalized_cycle_length = 0

                    distance = np.linalg.norm(np.array(pos[edge[0]]) - np.array(pos[edge[1]]))
                    if distance <= max_distance*0.3:
                        return None
                    normalized_distance = distance / max_distance

                    score = weight_factor * normalized_cycle_length + (1 - weight_factor) * (1 / normalized_distance)
                    return score  # Return the score if valid

            except nx.NetworkXNoCycle:
                return None  # No score for edges that do not complete a cycle
        return None  # No score for existing edges
    return None  # No score for intersecting edges

def compute_slack_metric(subgraph, node):
    degree = subgraph.degree(node)
    capacities = []
    for _, _, data in subgraph.edges(node, data=True):
        pandapower_data = data.get("pandapower_type", {})
        if not isinstance(pandapower_data, dict):
            pandapower_data = {}
        cap = pandapower_data.get("max_i_ka", 0.3)
        capacities.append(cap)
    avg_capacity = np.mean(capacities) if capacities else 0.3
    return degree * avg_capacity


def interpolate_failed_lines(net, failed_lines, nx_to_pp_bus_map, random_cable_data, line_sources, subgraph=None):
    """
    Interpolate failed lines by selecting appropriate line types from existing lines
    """
    logger.info("Interpolating failed lines")

    existing_line_types = []
    if len(net.line) > 0:
        line_params = net.line[["r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", 
                                "max_i_ka", "q_mm2", "alpha"]].copy()
        line_params["type"] = net.line.get("type", "cs")

        line_params = line_params.round(6)

        unique_params = line_params.drop_duplicates().to_dict('records')
        existing_line_types = unique_params
        
        logger.info(f"Found {len(existing_line_types)} unique line types in network")

    if not existing_line_types:
        logger.info("No existing line types found, using standard cable")
        for u, v in failed_lines:
            from_bus = nx_to_pp_bus_map[u]
            to_bus = nx_to_pp_bus_map[v]
            create_line_and_switch(net, from_bus, to_bus, random_cable_data, line_sources,
                                  line_type="standard_cable", line_name=f"{u}--{v}: interpolated->standard")
        return net, line_sources

    for u, v in failed_lines:
        from_bus = nx_to_pp_bus_map[u]
        to_bus = nx_to_pp_bus_map[v]
        selected_line_type = None
        if subgraph:
            adjacent_edge_types = []
            for node in [u, v]:
                for neighbor in subgraph.neighbors(node):
                    if subgraph.has_edge(node, neighbor):
                        edge_data = subgraph.edges[node, neighbor]
                        if "pandapower_type" in edge_data and isinstance(edge_data["pandapower_type"], dict):
                            adjacent_edge_types.append(edge_data["pandapower_type"])
            
            if adjacent_edge_types:
                selected_line_type = adjacent_edge_types[0]
                print(f"Using adjacent line type for {u}--{v}")
        
        if not selected_line_type:
            if len(existing_line_types) == 1:
                selected_line_type = existing_line_types[0]
            else:
                selected_line_type = max(existing_line_types, 
                                        key=lambda x: x.get("max_i_ka", 0))
            
            print(f"Using common line type for {u}--{v}")
        
        for param in ["r_ohm_per_km", "x_ohm_per_km"]:
            if selected_line_type[param] <= 1e-10:
                selected_line_type[param] = 0.01
                print(f"WARNING: Fixed zero {param} for line {u}--{v}")
        
        create_line_and_switch(net, from_bus, to_bus, selected_line_type, line_sources,
                              line_type="interpolated", line_name=f"{u}--{v}: interpolated->selected")
    
    return net, line_sources

def create_line_and_switch(net, from_bus, to_bus, line_params, line_sources, 
                                   line_type, line_name, is_switch=True):
    """
    Improved version that validates parameters before creating line.
    """
    try:
        # Validate and fix parameters
        fixed_params = validate_and_fix_cable_parameters(line_params)
        
        # Calculate line length from bus geodata
        if hasattr(net, 'bus_geodata') and from_bus in net.bus_geodata.index and to_bus in net.bus_geodata.index:
            dx = net.bus_geodata.loc[from_bus, "x"] - net.bus_geodata.loc[to_bus, "x"]
            dy = net.bus_geodata.loc[from_bus, "y"] - net.bus_geodata.loc[to_bus, "y"]
            length_km = max(np.sqrt(dx**2 + dy**2), 0.0001)
        else:
            length_km = 1.0  # Default 1 km
        
        # Create line
        line_index = pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=length_km,
            r_ohm_per_km=fixed_params["r_ohm_per_km"],
            x_ohm_per_km=fixed_params["x_ohm_per_km"],
            c_nf_per_km=fixed_params["c_nf_per_km"],
            max_i_ka=fixed_params["max_i_ka"],
            type=fixed_params.get("type", "cs"),
            q_mm2=fixed_params.get("q_mm2", 150),
            alpha=fixed_params.get("alpha", 0.004),
            name=line_name
        )
        
        line_sources[line_type] += 1
        
        # Create switch
        pp.create_switch(net, bus=from_bus, element=line_index, et="l", closed=is_switch)
        
        logger.debug(f"Successfully created {line_type} line: {line_name}")
        
    except Exception as e:
        logger.error(f"Failed to create line {line_name}: {e}")
        line_sources["failed"] += 1

def find_operating_voltage(subgraph):
    for _, _, data in subgraph.edges(data=True):
        if 'operatingvoltage' in data and not pd.isna(data['operatingvoltage']):
            if data["operatingvoltage"] > 1000:
                return data["operatingvoltage"] / 1000
            elif data["operatingvoltage"] < 1:
                return data["operatingvoltage"] * 1000
            else:
                return data['operatingvoltage']
    return 10

def scale_slack_buses(base_value, node_count, min_slack=1, max_slack=None):
    if max_slack is None:
        max_slack = min(int(node_count * 0.1), 20)

    min_slack = max(1, min_slack)
    max_slack = max(min_slack + 1, max_slack)
    
    small_graph = 50    
    if node_count <= small_graph:
        scale_factor = 1.0
    else:
        log_ratio = math.log(node_count / small_graph, 2) 
        scale_factor = 1.0 + (log_ratio * 0.5)  #
    scaled_value = int(base_value * scale_factor)

    noise = random.uniform(-0.3, 0.3)
    final_value = int(scaled_value + noise * scaled_value)

    final_value = max(min_slack, min(final_value, max_slack))
    
    return final_value

def select_top_slack_nodes(subgraph, candidate_nodes, num_slack):
    min_distance = len(subgraph.edges)**0.5 / 2.0
    metrics = {node: compute_slack_metric(subgraph, node) for node in candidate_nodes}
    
    sorted_nodes = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
    
    selected_slack_nodes = []
    considered_nodes = []

    for node, metric in sorted_nodes:
        considered_nodes.append(node)
        too_close = False
        for selected_node in selected_slack_nodes:
            try:
                distance = nx.shortest_path_length(subgraph, node, selected_node)
                if distance < min_distance:
                    too_close = True
                    break
            except nx.NetworkXNoPath:
                pass
        
        if not too_close:
            selected_slack_nodes.append(node)
            logger.info(f"Selected slack node {node} with metric {metric:.4f}")
        if len(selected_slack_nodes) >= num_slack:
            break
    if len(selected_slack_nodes) < num_slack:
        logger.info(f"Could only select {len(selected_slack_nodes)} nodes with spacing constraint, filling remaining {num_slack - len(selected_slack_nodes)}")
        remaining_candidates = [node for node, _ in sorted_nodes if node not in selected_slack_nodes]
        additional_nodes = remaining_candidates[:min(num_slack - len(selected_slack_nodes), len(remaining_candidates))]
        
        for node in additional_nodes:
            selected_slack_nodes.append(node)
            logger.info(f"Added additional slack node {node} with metric {metrics[node]}")
    return selected_slack_nodes

def validate_and_fix_cable_parameters(cable_data: dict) -> dict:
    """
    Enhanced validation that aggressively fixes impedance issues
    """
    fixed_data = cable_data.copy()
    
    r_ohm = fixed_data.get('r_ohm_per_km', 0.1)
    x_ohm = fixed_data.get('x_ohm_per_km', 0.1)
    max_i_ka = fixed_data.get('max_i_ka', 0.3)
    q_mm2 = fixed_data.get('q_mm2', 150)

    # Define minimum acceptable values (increased from 0.001)
    MIN_IMPEDANCE = 0.10  # 10 mΩ/km minimum
    TARGET_MIN_IMPEDANCE = 0.20  # 20 mΩ/km preferred minimum
    
    # Fix extremely low resistance values
    if r_ohm <= MIN_IMPEDANCE:
        multiplier = 1
        temp_r = r_ohm
        while temp_r <= TARGET_MIN_IMPEDANCE and multiplier <= 1000:
            temp_r *= 10
            multiplier *= 10
        fixed_data['r_ohm_per_km'] = temp_r
        logger.warning(f"Fixed R from {r_ohm:.6f} to {temp_r:.6f} Ω/km (multiplied by {multiplier})")
    
    # Fix extremely low reactance values
    if x_ohm <= MIN_IMPEDANCE:
        multiplier = 1
        temp_x = x_ohm
        while temp_x <= TARGET_MIN_IMPEDANCE and multiplier <= 1000:
            temp_x *= 10
            multiplier *= 10
        fixed_data['x_ohm_per_km'] = temp_x
        logger.warning(f"Fixed X from {x_ohm:.6f} to {temp_x:.6f} Ω/km (multiplied by {multiplier})")
    
    # Check R/X ratio after fixes
    new_r = fixed_data['r_ohm_per_km']
    new_x = fixed_data['x_ohm_per_km']
    rx_ratio = new_r / new_x if new_x > 0 else float('inf')
    
    if rx_ratio > 8:  
        logger.warning(f"High R/X ratio: {rx_ratio:.2f}. Adjusting X upward.")
        fixed_data['x_ohm_per_km'] = new_r / 3.0  
    elif rx_ratio < 0.1:  
        logger.warning(f"Low R/X ratio: {rx_ratio:.2f}. Adjusting R upward.")
        fixed_data['r_ohm_per_km'] = new_x * 0.3  
    
    # Validate current rating
    expected_current_range = (q_mm2 * 0.5, q_mm2 * 3.0)
    current_in_amps = max_i_ka * 1000
    
    if current_in_amps < expected_current_range[0]:
        suggested_current = q_mm2 * 1.5 / 1000
        logger.warning(f"Current rating {max_i_ka:.3f} kA too low for {q_mm2} mm² cable")
        fixed_data['max_i_ka'] = suggested_current
    
    # Ensure absolute minimums (safety net)
    fixed_data['r_ohm_per_km'] = max(fixed_data['r_ohm_per_km'], MIN_IMPEDANCE)
    fixed_data['x_ohm_per_km'] = max(fixed_data['x_ohm_per_km'], MIN_IMPEDANCE)
    
    # Set defaults for missing parameters
    fixed_data['c_nf_per_km'] = fixed_data.get('c_nf_per_km', 200)
    fixed_data['alpha'] = fixed_data.get('alpha', 0.004)
    fixed_data['type'] = fixed_data.get('type', 'cs')
    
    return fixed_data

def sanitize_edge_data(edge_data: dict, dist_callable, existing_line_types) -> dict:
    if not edge_data:
        logger.debug("No edge data provided, using standard cable")
        return STANDARD_CABLE_MAPPING[dist_callable["standard_cables"]()].copy()

    cleaned = edge_data.copy()

    required_params = ["r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "max_i_ka", "q_mm2", "alpha"]
    
    for param in required_params:
        if param not in cleaned or cleaned[param] == 0:
            default_cable = STANDARD_CABLE_MAPPING[dist_callable["standard_cables"]()]
            logger.warning(f"Missing or zero value for {param}. Using default: {default_cable[param]}")
            cleaned[param] = default_cable[param]
    cleaned = validate_and_fix_cable_parameters(cleaned)
    return cleaned


def analyze_network_issues(net, info):
    logger.info("=== NETWORK ANALYSIS ===")
    
    # Check for overloaded lines
    if not net.res_line.empty:
        overloaded = net.res_line[net.res_line.loading_percent > 100]
        if len(overloaded) > 0:
            logger.warning(f"Found {len(overloaded)} overloaded lines (>{len(overloaded)/len(net.res_line)*100:.1f}% of total)")
            worst_lines = overloaded.nlargest(5, 'loading_percent')
            for idx, row in worst_lines.iterrows():
                logger.warning(f"Line {idx}: {row.loading_percent:.1f}% loaded, {row.pl_mw*1000:.1f}W losses")
        
        # Check for very lightly loaded lines
        light_loaded = net.res_line[net.res_line.loading_percent < 1.0]
        if len(light_loaded) > 0:
            logger.info(f"Found {len(light_loaded)} very lightly loaded lines (<1%)")
    
    # Check voltage issues
    if not net.res_bus.empty:
        voltage_issues = net.res_bus[(net.res_bus.vm_pu < 0.95) | (net.res_bus.vm_pu > 1.05)]
        if len(voltage_issues) > 0:
            logger.warning(f"Found {len(voltage_issues)} buses with voltage issues")
    
    # Check generation/load balance
    total_gen = info.get('total_active_generation', 0)
    total_load = info.get('total_active_load', 0)
    if total_load > 0:
        gen_load_ratio = total_gen / total_load
        if gen_load_ratio > 3:
            logger.warning(f"Very high generation/load ratio: {gen_load_ratio:.2f}")
        elif gen_load_ratio < 0.8:
            logger.warning(f"Low generation/load ratio: {gen_load_ratio:.2f}")
def preprocess_edge_impedances(subgraph):
    """
    Preprocess all edge impedances before creating PandaPower network
    """
    fixed_edges = 0
    for u, v, edge_data in subgraph.edges(data=True):
        pandapower_data = edge_data.get("pandapower_type", {})
        if isinstance(pandapower_data, dict) and pandapower_data:
            original_r = pandapower_data.get('r_ohm_per_km', 0)
            original_x = pandapower_data.get('x_ohm_per_km', 0)
            
            # Apply fixes
            fixed_params = validate_and_fix_cable_parameters(pandapower_data)
            
            # Update edge data if changes were made
            if (fixed_params['r_ohm_per_km'] != original_r or 
                fixed_params['x_ohm_per_km'] != original_x):
                edge_data["pandapower_type"] = fixed_params
                fixed_edges += 1
                logger.debug(f"Fixed impedances for edge {u}-{v}")
    
    logger.info(f"Preprocessed {fixed_edges} edges with impedance issues")
    return subgraph
def create_pandapower_network(subgraph: nx.Graph, args: dict, dist_callable: dict) -> pp.pandapowerNet:
    subgraph = preprocess_edge_impedances(subgraph)
    net = pp.create_empty_network()

    base_num_slack = dist_callable["n_slack_busses"]()
    num_slack = scale_slack_buses(base_num_slack, len(subgraph.nodes))
    
    logger.info(f"Graph has {len(subgraph.nodes)} nodes. Base slack buses: {base_num_slack}, Scaled slack buses: {num_slack}")
    
    vn_kv = find_operating_voltage(subgraph)
    if vn_kv != 10.0: 
        voltage_correction = vn_kv / 10.0
        for node in subgraph.nodes:
            if "net_load" in subgraph.nodes[node]:
                subgraph.nodes[node]["net_load"] *= voltage_correction

    logger.info(f"Operating voltage: {vn_kv} kV")
    
    nx_to_pp_bus_map = {}
    candidate_nodes = []
    remaining_nodes = []
    line_sources = {"original_data": 0, "interpolated": 0, "standard_cable": 0}
    
    for node, data in subgraph.nodes(data=True):
        candidate_nodes.append(node)
        remaining_nodes.append(node)
        bus = pp.create_bus(net, vn_kv=vn_kv, geodata=(data["geometry"].x, data["geometry"].y), 
                            coords=(data["geometry"].x, data["geometry"].y))
        nx_to_pp_bus_map[node] = bus
    net["nx_to_pp_bus_map"] = nx_to_pp_bus_map

    selected_slack_nodes = select_top_slack_nodes(subgraph, candidate_nodes, num_slack)
    total_load = sum(data.get("net_load", 0) for node, data in subgraph.nodes(data=True) if data.get("net_load", 0) > 0)
    slack_power = total_load / len(selected_slack_nodes) if selected_slack_nodes else 0
    for slack_node in selected_slack_nodes:
        logger.info(f"Creating slack generator for {slack_node}")
        bus_id = nx_to_pp_bus_map[slack_node]
        vm_set = dist_callable["slack_vm_set"]()
        pp.create_gen(net, slack=True, bus=bus_id, vm_pu=vm_set, p_mw=slack_power, min_q_mvar=-0.1, max_q_mvar=0.1)
        net.bus.loc[bus_id, "name"] = f"{slack_node}_Slack"
        if slack_node in remaining_nodes:
            remaining_nodes.remove(slack_node)
    
    for node, data in subgraph.nodes(data=True):
        if node in remaining_nodes:
            bus_id = nx_to_pp_bus_map[node]
            net_load = data.get("net_load", 0)
            power_factor = data.get("power_factor", 0.9)
            
            if net_load > 0:
                # Calculate reactive power based on power factor
                q_mvar = net_load * np.tan(np.arccos(power_factor))
                pp.create_load(net, bus=bus_id, p_mw=net_load, q_mvar=q_mvar,
                             scaling=1.0, controllable=False)
                net.bus.loc[bus_id, "name"] = f"{node}_Load"
            elif net_load < 0:
                # For generators, assume better power factor
                pp.create_gen(net, bus=bus_id, vm_pu=1.01, p_mw=abs(net_load),
                            min_p_mw=0, max_p_mw=abs(net_load)*1.2)
                net.bus.loc[bus_id, "name"] = f"{node}_DG"
            else:
                net.bus.loc[bus_id, "name"] = f"{node}_NoLoad"
                
  # Collect existing line types BEFORE processing edges
    existing_line_types = []
    complete_line_count = 0
    
    for u, v, edge_data in subgraph.edges(data=True):
        logger.debug(f"Processing edge {u}--{v} with data: {edge_data}")
        pandapower_data = edge_data.get("pandapower_type", {})
        if isinstance(pandapower_data, dict) and pandapower_data:
            required_params = ["r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "max_i_ka", "q_mm2", "alpha"]
            if all(pandapower_data.get(param, 0) > 0 for param in required_params):
                existing_line_types.append(pandapower_data)
                complete_line_count += 1

    logger.info(f"Found {len(existing_line_types)} complete line types from {complete_line_count} edges in original network")
    
    # Initialize line_sources properly
    line_sources = {"original_data": 0, "interpolated": 0, "standard_cable": 0, "failed": 0}
    failed_lines = []
    
    # Process each edge with proper categorization
    for u, v, edge_data in subgraph.edges(data=True):
        from_bus = nx_to_pp_bus_map[u]
        to_bus = nx_to_pp_bus_map[v]
        
        # Determine line type and source
        pandapower_data = edge_data.get("pandapower_type", {})
        if isinstance(pandapower_data, dict) and pandapower_data:
            required_params = ["r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "max_i_ka", "q_mm2", "alpha"]
            if all(pandapower_data.get(param, 0) > 0 for param in required_params):
                # Complete original data
                line_data = pandapower_data
                source_type = "original_data"
                logger.debug(f"Using complete original data for edge {u}--{v}")
            else:
                # Incomplete data - needs interpolation
                line_data = sanitize_edge_data(pandapower_data, dist_callable, existing_line_types)
                if existing_line_types:
                    source_type = "interpolated"
                    logger.debug(f"Using interpolated data for edge {u}--{v}")
                else:
                    source_type = "standard_cable"
                    logger.debug(f"Using standard cable for edge {u}--{v}")
        else:
            # No pandapower data at all
            line_data = sanitize_edge_data({}, dist_callable, existing_line_types)
            source_type = "standard_cable"
            logger.debug(f"No pandapower data for edge {u}--{v}, using standard cable")
        
        label = edge_data.get("label", "Unknown")
        is_switch = not edge_data.get("is_switch", False)
        
        try:
            create_line_and_switch(net, from_bus, to_bus, line_data, line_sources,
                                 line_type=source_type, line_name=f"{u}--{v}:{label}",
                                 is_switch=is_switch)
        except Exception as e:
            logger.warning(f"Failed to create line {u}--{v}: {e}")
            failed_lines.append((u, v))

    # Calculate and log success rate
    total_edges = len(subgraph.edges)
    successful_lines = sum(line_sources[k] for k in ["original_data", "interpolated", "standard_cable"])
    success_rate = (successful_lines / total_edges) * 100 if total_edges > 0 else 0
    
    logger.info(f"Line creation success rate: {success_rate:.1f}% ({successful_lines}/{total_edges})")
    logger.info(f"Line sources breakdown: {line_sources}")
   
    net.gen.loc[net.gen.slack, 'slack_weight'] = 1.0/len(net.gen[net.gen.slack])   
    
    add_reactive_compensation(net)
    converged =net.converged
    try:
        pp.runpp(net, max_iteration=100, v_debug=True, run_control=True, initialization="dc", 
                 calculate_voltage_angles=True)
        converged = net.converged
        logger.info("Initial power flow calculation completed.")
    except Exception as e:
        err_msg = str(e)
        logger.warning(f"Initial power flow failed: {err_msg}")
        net.converged = False

    r_ohm = net.line["r_ohm_per_km"]
    x_ohm = net.line["x_ohm_per_km"]
    
     # Handle empty series and make sure there are values before taking min
    if not r_ohm.empty and not x_ohm.empty:
        min_r = r_ohm.min()
        min_x = x_ohm.min()
        logger.info(f"Minimum r and x values: {min_r}, {min_x}")
    else:
        logger.info("No resistance or reactance values found")


    if not converged:
        logger.info("Power flow calculation failed after multiple attempts.")
        logger.info("Adding additional slack generator to improve convergence.")
        non_slack_nodes = [node for node in candidate_nodes if node not in selected_slack_nodes]
        if non_slack_nodes:
            non_slack_metrics = {node: compute_slack_metric(subgraph, node) for node in non_slack_nodes}
            max_non_slack = max(non_slack_metrics.values())
            additional_candidates = [node for node, m in non_slack_metrics.items() if m == max_non_slack]
            new_slack = random.choice(additional_candidates)
            bus_id = nx_to_pp_bus_map[new_slack]
            extra_load = sum(net.load.p_mw) if not net.load.empty else 0
            pp.create_gen(net, slack=True, bus=bus_id, vm_pu=1.02, p_mw=extra_load)
            net.bus.loc[bus_id, "name"] = f"{new_slack}_ExtraSlack"
            logger.info(f"Added additional slack bus at node {new_slack}")
            try:
                pp.runpp(net, max_iteration=100, v_debug=True, run_control=True, initialization="dc",
                         calculate_voltage_angles=True)
                if net.converged:
                    converged = True
                    logger.info("Successfully ran PP network after adding extra slack.")
            except Exception as e:
                logger.warning(f"Final attempt after adding extra slack failed: {e}")

    try:
        pp.runpp(net, max_iteration=100, v_debug=True, run_control=True, initialization="dc", 
                 calculate_voltage_angles=True)
        logger.info("SUCCESFULLY RAN PP NETWORK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    except Exception as e:
        err_msg = str(e)
        logger.warning(f"Initial power flow failed: {err_msg}")
        net.converged = False
        match = re.search(r"Distributed slack contribution factors in island '([^']+)'", err_msg)
        if match:
            island_nodes_str = match.group(1).strip("[]")
            island_nodes = [s.strip() for s in island_nodes_str.split(',') if s.strip() != '']
            try:
                island_nodes = [int(x) for x in island_nodes]
            except Exception as e:
                logger.warning(f"Error converting island node strings to integers: {e}")
    if args.show_pandapower_report:
        result = pp.diagnostic(net, detailed_report= True,warnings_only=False)
        logger.info(result)

    info = {
        "num_buses": len(net.bus),
        "num_lines": len(net.line),
        "num_switches": len(net.switch),
        "average_load": net.res_load.p_mw.mean() if not net.res_load.empty else None,
        "average_gen": net.res_gen.p_mw.mean() if not net.res_gen.empty else None,
        "average_line_utilization": net.res_line.loading_percent.mean() if not net.res_line.empty else None,
        "average_line_losses": net.res_line.pl_mw.mean() if not net.res_line.empty else None,
        "max_line_utilization": net.res_line.loading_percent.max() if not net.res_line.empty else None,
        "min_line_utilization": net.res_line.loading_percent.min() if not net.res_line.empty else None,
        "max_line_losses": net.res_line.pl_mw.max() if not net.res_line.empty else None,
        "min_line_losses": net.res_line.pl_mw.min() if not net.res_line.empty else None,
        "max_bus_voltage": net.res_bus.vm_pu.max() if not net.res_bus.empty else None,
        "min_bus_voltage": net.res_bus.vm_pu.min() if not net.res_bus.empty else None,
        "std_bus_voltage": net.res_bus.vm_pu.std() if not net.res_bus.empty else None,
        "total_active_load": net.res_load.p_mw.sum() if not net.res_load.empty else None,
        "total_reactive_load": net.res_load.q_mvar.sum() if not net.res_load.empty else None,
        "total_active_generation": net.res_gen.p_mw.sum() if not net.res_gen.empty else None,
        "total_reactive_generation": net.res_gen.q_mvar.sum() if not net.res_gen.empty else None,
        "slack_power_p_mw": net.res_gen[net.gen.slack == True].p_mw.sum() if not net.res_gen.empty else None,
        "slack_power_q_mvar": net.res_gen[net.gen.slack == True].q_mvar.sum() if not net.res_gen.empty else None,
        "percent_closed_switches": (net.switch.closed.sum() / len(net.switch)) * 100 if len(net.switch) > 0 else 0,
        "gen_to_load_ratio": net.res_gen.p_mw.sum() / max(net.res_load.p_mw.sum(), 1e-9) if (not net.res_gen.empty and not net.res_load.empty) else None,
        "num_overloaded_lines": (net.res_line.loading_percent > 100).sum() if not net.res_line.empty else None,
        "failed_lines_count": len(failed_lines),
        "failed_lines_ratio": len(failed_lines) / len(list(subgraph.edges)) if len(list(subgraph.edges)) > 0 else None,
    }
    logger.info(info)
    analyze_network_issues(net, info)
    
    return net, info 

def add_voltage_regulators(net):
    slack_buses = net.gen[net.gen.slack == True].bus.values
    if len(slack_buses) > 0 and len(net.bus) > 20:
        candidate_buses = []
        
        for bus in net.bus.index:
            if bus not in slack_buses:
                # Check if bus is suitable for transformer
                connected_lines = len(net.line[(net.line.from_bus == bus) | 
                                              (net.line.to_bus == bus)])
                if connected_lines >= 3:  # Junction point
                    candidate_buses.append(bus)
      
        num_trafos = min(2, len(candidate_buses))
        selected_buses = random.sample(candidate_buses, num_trafos) if candidate_buses else []
        
        for bus in selected_buses:
            vn_kv= net.bus.at[bus, "vn_kv"]
            new_bus = pp.create_bus(
                net,
                vn_kv= vn_kv* 0.95,
                geodata=(net.bus_geodata.at[bus, "x"],  net.bus_geodata.at[bus, "y"]))
            # Add transformer
            pp.create_transformer_from_parameters(
                net, hv_bus=bus, lv_bus=new_bus,
                sn_mva=10, vn_hv_kv=vn_kv, vn_lv_kv=vn_kv*0.95,
                vkr_percent=0.5, vk_percent=4, pfe_kw=0, i0_percent=0,
                tap_pos=0, tap_neutral=0, tap_min=-2, tap_max=2,
                tap_step_percent=2.5
            )
            
            # Move some loads to the new bus
            loads_to_move = net.load[net.load.bus == bus].index[:len(net.load)//4]
            for load_idx in loads_to_move:
                net.load.at[load_idx, "bus"] = new_bus

def process_single_subgraph(graph_id, subgraph, dfs, args, dist_callables, save_location):
    logger.debug("inside process_single_subgraph")
    suffixes = list(string.ascii_lowercase)
    succ = fail = 0

    original_subgraph = copy.deepcopy(subgraph)
    for sampled_idx in range(args.n_samples_per_graph):
        timeframes = sample_timeframes(args.n_loadcase_time_intervals, args.interval_duration_minutes)
        date_time = timeframes[0]
        suffix = suffixes[sampled_idx % len(suffixes)]
        logger.info(f"Modifying subgraph {graph_id} for sample {sampled_idx + 1}")
        base = copy.deepcopy(original_subgraph)
        modified_subgraph = modify_subgraph(base, args, dist_callables)

        modified_subgraph = assign_realistic_loads(
            modified_subgraph, args, dist_callables, timeframes, dfs
        )

        electrified_network, info = create_pandapower_network(modified_subgraph, args, dist_callables)
        # Log the line loading statistics
        if electrified_network.converged and info.get('max_line_loading') is not None:
            logger.info(f"Max line loading: {info['max_line_loading']:.1f}%, "
                       f"Mean: {info['mean_line_loading']:.1f}%")
            
            # Check if any line exceeds threshold
            if info['max_line_loading'] > args.max_line_loading:
                logger.warning(f"Line loading exceeds threshold: {info['max_line_loading']:.1f}% > {args.max_line_loading}%")
        
        if electrified_network.converged:
            succ += 1
            name = f"{graph_id}_mod_{suffix}"
            save_single_graph(name, electrified_network, save_location)
            logger.info(f"[Worker {os.getpid()}]  Converged - about to save {graph_id}")
        else:
            fail += 1
            logger.info(f"Power flow failed for {graph_id}@{date_time}")
            logger.info(f"[Worker {os.getpid()}]  Did *not* converge for {graph_id}@{date_time}")
                            
    logger.info(f"[Worker {os.getpid()}] Finished processing subgraph {graph_id} with {succ} successes and {fail} failures.")
    return succ,fail

def get_n_workers():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)) # Fallback to 1 if cpu_count is not implemented    

def _worker(graph_id, subgraph, dfs, args, dist_callables, save_location):
    """Enhanced worker with better error handling and cleanup"""
    pid = os.getpid()
    try:
        # Set up process-specific random seeds
        random.seed()
        np.random.seed(int.from_bytes(os.urandom(4), 'little'))
        
        # Set up minimal logging for this process
        logging.getLogger("data_generation")
        logger.setLevel(logging.DEBUG)
        
        logger.info(f"[Worker {pid}] Processing subgraph {graph_id}, nodes={len(subgraph.nodes)}")
        
        result = process_single_subgraph(graph_id, subgraph, dfs, args, dist_callables, save_location)
        logger.info(f"[Worker {pid}] Completed subgraph {graph_id}")
        return result
        
    except Exception as e:
        logger.error(f"[Worker {pid}] Failed processing subgraph {graph_id}: {e}")
        return 0, 1  # 0 success, 1 failure
    finally:
        # Cleanup logging handlers to prevent conflicts
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)
def transform_subgraphs(
    distributions: Dict[str, Any],
    dfs: Any,
    args: Any,
) -> Dict[str, int]:
   
    # initialize distributions
    dist_callables = initialize_distributions(distributions)

    # prepare save location
    save_dir = Path(args.save_dir or os.getcwd())
    date_str = datetime.now().strftime("%d%m%Y")
    mode_str = "all" if args.iterate_all else f"range-{args.target_busses}-{args.bus_range}"
    save_name = f"_synthetic-train-data_{date_str}_{mode_str}_{args.num_subgraphs}"
    save_location = save_dir / save_name / "original"
    save_location.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {save_location}")

    pp_dir = save_location / "pandapower_networks"
    pp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("start loading subgraphs")

    # load all subgraphs once
    raw_subgraphs = load_subgraphs(args)
    if not raw_subgraphs:
        raise RuntimeError("No subgraphs found to process.")

    node_range = [len(sg.nodes) for sg in raw_subgraphs]
    logger.info(f"Subgraph node ranges for {len(node_range)} graphs: {node_range}")

    total_required = len(raw_subgraphs) if args.iterate_all else args.num_subgraphs
    logger.info(f"\n\n start transforming subgraphs need to process {total_required} \n \n ")

    # Determine which subgraphs to process
    if args.iterate_all:
        to_do = [(i, sg) for i, sg in enumerate(raw_subgraphs)]
    else:
        sampled_subgraphs = random.choices(raw_subgraphs, k=args.num_subgraphs)
        to_do = [(i, sg) for i, sg in enumerate(sampled_subgraphs)]

    logger.info(f"Will process {len(to_do)} jobs from {len(raw_subgraphs)} available subgraphs")

    processed = {"total": 0, "successful": 0, "failed": 0}
    pbar = tqdm(total=len(to_do), desc="Processing subgraphs", unit="subgraph")

    if args.multiprocessing:
        logger.info("Using multiprocessing with ProcessPoolExecutor")
        with ProcessPoolExecutor(max_workers=get_n_workers(), initializer=setup_logging) as ex:
            futures = [
                ex.submit(_worker, graph_id, subgraph, dfs, args, dist_callables, save_location)
                for graph_id, subgraph in to_do
            ]

            for i, fut in enumerate(as_completed(futures), start=1):
                try:
                    suc, fl = fut.result(timeout=180)
                    processed["successful"] += suc
                    processed["failed"] += fl
                    processed["total"] += (suc + fl)
                    logger.debug(f"Job {i} completed: {suc} success, {fl} failed")
                except Exception as e:
                    if isinstance(e, concurrent.futures.TimeoutError):
                        logger.warning(f"Job {i} timed out after 3 minutes")
                    else:
                        logger.error(f"Job {i} failed: {e}")
                    processed["failed"] += 1
                    processed["total"] += 1
                finally:
                    pbar.update(1)

        pbar.close()
        for bar in list(tqdm._instances):
            bar.close()

        logger.info(f"Finished: {processed}")
    else:
        logger.info("Using single-threaded processing")
        for graph_id, subgraph in to_do:
            try:
                suc, fl = _worker(graph_id, subgraph, dfs, args, dist_callables, save_location)
                processed["successful"] += suc
                processed["failed"] += fl
                processed["total"] += (suc + fl)
            
            except Exception as e:
                logger.error(f"Job {graph_id} failed: {e}")
                processed["failed"] += 1
                processed["total"] += 1
            finally:
                pbar.update(1)

        pbar.close()

        logger.info(f"Finished: {processed}")

    return processed


def save_single_graph(graph_name, pp_network, save_location):
    
    pp_dir = save_location/  "pandapower_networks"
    try:
        pp.runpp(pp_network, max_iteration=100, v_debug=False, run_control=True, initialization="dc", calculate_voltage_angles=True)
    except Exception as e:
        logger.warning(f"Power flow did not converge for {graph_name}: {e}")
    
    pp_file = os.path.join(pp_dir, f"{graph_name}.json")
    with open(pp_file, "w") as f:
        json.dump(pp.to_json(pp_network), f)

    logger.info(f"Saved graph {graph_name} to {save_location}")
    logger.info(f"[Worker {os.getpid()}] Saving graph {graph_name} to {pp_dir}")

