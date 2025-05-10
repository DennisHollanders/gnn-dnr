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
import uuid

random.seed(0)
logger = logging.getLogger(__name__)

# # Ensure subgraph nodes have valid positions
# def get_subgraph_centroid(subgraph):
#     positions = [
#         node_data["position"] for _, node_data in subgraph.nodes(data=True)
#         if "position" in node_data and isinstance(node_data["position"], (list, tuple)) and len(node_data["position"]) == 2
#     ]
#     if not positions:
#         raise ValueError("No valid positions found in the subgraph nodes.")
#     avg_pos = np.mean(positions, axis=0)
#     return Point(avg_pos)

# # Find the nearest postcode6 for a given subgraph centroid
# def find_closest_postcode6(cbs_pc6_gdf, subgraph):
#     center = get_subgraph_centroid(subgraph)
#     center_geo = gpd.GeoSeries([center], crs="EPSG:28992")  # Assume centroid is in EPSG:28992
#     # Ensure CRS matches
#     center_geo_projected = center_geo.to_crs(cbs_pc6_gdf.crs)
#     # Calculate distances
#     cbs_pc6_gdf["distance"] = cbs_pc6_gdf.geometry.distance(center_geo_projected.iloc[0])
#     closest_row = cbs_pc6_gdf.loc[cbs_pc6_gdf["distance"].idxmin()]
#     return closest_row["postcode6"]

# # Match postcode6 to the corresponding buurtcode
# def match_postcode6_to_buurt(postcode6, buurt_to_postcodes):
#     match = buurt_to_postcodes.loc[buurt_to_postcodes["postcode6"] == postcode6, "buurtcode"]
#     if not match.empty:
#         return match.iloc[0]
#     return None

# # Find both the postcode6 and the corresponding buurtcode
# def find_postcode6_and_buurt(cbs_pc6_gdf, buurt_to_postcodes, subgraph):
#     closest_postcode6 = find_closest_postcode6(cbs_pc6_gdf, subgraph)
#     matching_buurt = match_postcode6_to_buurt(closest_postcode6, buurt_to_postcodes)
#     return closest_postcode6, matching_buurt

# # Process multiple subgraphs
# def find_postcode6s_and_buurts(subgraphs,dfs):
#     cbs_pc6_gdf, buurt_to_postcodes = dfs[1], dfs[2]

#     buurt_to_postcodes["postcode6"] = buurt_to_postcodes["postcode6"].apply(
#         lambda x: ast.literal_eval(x) if isinstance(x, str) else x
#     )
#     expanded_buurt_to_postcodes = buurt_to_postcodes.explode("postcode6").reset_index(drop=True)

#     # Process each subgraph
#     results = []
#     for sg in subgraphs:
#         try:
#             postcode6, buurt = find_postcode6_and_buurt(cbs_pc6_gdf, expanded_buurt_to_postcodes, sg)
#             print(f"Closest postcode6: {postcode6}, CBS Buurt: {buurt}")
#             results.append({"postcode6": postcode6, "buurt": buurt})
#         except ValueError as e:
#             print(f"Error: {e}")
#             results.append({"postcode6": None, "buurt": None})

#     return pd.DataFrame(results)

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
    logger.info(f"Sampled {dist_name} -> {sample}")
    return sample

def initialize_distributions(distributions: Dict[str, dict]) -> Dict[str, Any]:
    """
    Returns a dict of dist_name -> callable() for sampling.
    """
    return {
        name: partial(sample_from_distribution_global, name, params)
        for name, params in distributions.items()
    }

def retrieve_standard_production(timestamp,df):
    timestamp = pd.to_datetime(timestamp)
    A_columns = [col for col in df.columns if col.endswith("_A")]
    A_col = random.choice(A_columns)
    I_col = A_col.rsplit("_A",1)[0] +"_I"
    row = df[pd.to_datetime(df["from"]) == timestamp]
    consumption = row[A_col].values[0]
    production = row[I_col].values[0]
    max_consumption = row[A_col].values.sum()
    max_production = row[I_col].values.sum()
    return consumption,production, max_consumption,max_production

def sample_timeframes(num_intervals, interval_duration_minutes=15):
    start_of_year = pd.Timestamp("2025-01-01 00:00:00")
    end_of_year = pd.Timestamp("2025-12-31 23:59:59")
    total_minutes = (end_of_year - start_of_year).total_seconds() / 60
    max_intervals = int(total_minutes / interval_duration_minutes)
    random_start_interval = random.randint(0, max_intervals - num_intervals)
    random_start_time = start_of_year + timedelta(minutes=random_start_interval * interval_duration_minutes)
    timeframes = [ random_start_time + timedelta(minutes=i * interval_duration_minutes) for i in range(num_intervals)]
    
    return timeframes

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
    """
    Load only those .pkl files whose embedded node-range overlaps with args.node_range.
    Filenames include a numeric range string (e.g., '130140' means 130 to 140).
    """
    folder_path = args.subgraph_folder
    print(f"Loading from folder: {folder_path}")
    file_names = os.listdir(folder_path)

    # simple args holder

    subgraphs = []

    if args.iterate_all:
        print("Loading all .pkl files")
        for file_name in tqdm(file_names, desc="Loading files", unit="file"):
            full_path = os.path.join(folder_path, fname)
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
        print("Filtering by args.node_range", bus_range_min, " -  ", bus_range_max)
        for fname in tqdm(file_names, desc= "Loading files", unit="file"):
            # extract first numeric token
            nums = re.findall(r"(\d+)", fname)
            if not nums:
                continue
            token = nums[0]
            half = len(token) // 2
            lower = int(token[:half])
            upper = int(token[half:])
            print(f"File {fname}: range {lower}-{upper}")
            # check overlap
            if lower <= bus_range_max and upper >= bus_range_min:
                full_path = os.path.join(folder_path, fname)
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
            print("Added an edge")
            n_cycles += 1
            order_added += 1
        else:
            convex_layers_to_use = list(set([layer + 1 for layer in convex_layers_to_use]) | set([layer - 1 for layer in convex_layers_to_use]) | {0})
            failed_attempts += 1
            if failed_attempts > 10:
                print("Failed to add edges. Stopping.")
                break

    logger.info(f"cycles_added: {order_added}")
    logger.info(f"convex_layers_used: {convex_layers_to_use}")
    if args.plot_added_edge:
        fig = plt.figure(figsize=(12, 6))
        pos = nx.get_node_attributes(subgraph_to_adapt, 'position')
        # Draw all nodes in black
        nx.draw_networkx_nodes(subgraph_to_adapt, pos, node_color='black', node_size=10)
        # Draw all edges in black
        nx.draw_networkx_edges(subgraph_to_adapt, pos, edge_color='black')
        # Identify synthetic edges
        synthetic_edges = [(u, v) for u, v, d in subgraph_to_adapt.edges(data=True) if d.get('is_synthetic', False)]
        if synthetic_edges:
            # Draw synthetic edges in red
            nx.draw_networkx_edges(subgraph_to_adapt, pos, edgelist=synthetic_edges, edge_color='red', width=2)
            # Identify nodes connected by synthetic edges and draw them in red
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
        return longest_paths[0] + longest_paths[1] + 1  # Connecting the two longest paths
    elif len(longest_paths) == 1:
        return longest_paths[0] + 1  # longest path plus one edge
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
    # for i in range(len(selected_nodes)):
    #     for j in range(i + 1, len(selected_nodes)):
    #         u, v = selected_nodes[i], selected_nodes[j]
    #         if subgraph_to_adapt.has_edge(u, v):
    #             continue  
    #         distance = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
    #         if distance < args.min_distance_threshold:
    #             continue  
    #         edge = (u, v)
    #         score = check_and_add_edge(edge, subgraph_to_adapt, pos, max_distance, max_cycle_length, args.weight_factor)
    #         if score is not None:
    #             possible_edges.append((edge, score))

    possible_edges.sort(key=lambda x: x[1], reverse=True)

    if not possible_edges:
        print("No valid edges to add.")
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
        geodata=[pos[best_edge[0]], pos[best_edge[1]]]
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
    rather than averaging parameters.
    """
    print("Interpolating failed lines with improved method.")
    
    # First collect all existing line types in the network
    existing_line_types = []
    if len(net.line) > 0:
        # Group lines by their electrical parameters to identify distinct types
        line_params = net.line[["r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", 
                                "max_i_ka", "q_mm2", "alpha"]].copy()
        line_params["type"] = net.line.get("type", "cs")
        
        # Round slightly to handle floating point variations
        line_params = line_params.round(6)
        
        # Get unique parameter combinations
        unique_params = line_params.drop_duplicates().to_dict('records')
        existing_line_types = unique_params
        
        print(f"Found {len(existing_line_types)} unique line types in network")
    
    # If no existing line types, use the provided standard cable
    if not existing_line_types:
        print("No existing line types found, using standard cable")
        for u, v in failed_lines:
            from_bus = nx_to_pp_bus_map[u]
            to_bus = nx_to_pp_bus_map[v]
            create_line_and_switch(net, from_bus, to_bus, random_cable_data, line_sources,
                                  line_type="standard_cable", line_name=f"{u}--{v}: interpolated->standard")
        return net, line_sources
    
    # For each failed line, find the most appropriate line type
    for u, v in failed_lines:
        from_bus = nx_to_pp_bus_map[u]
        to_bus = nx_to_pp_bus_map[v]
        
        selected_line_type = None
        
        # APPROACH 1: Check if we can find line types connected to either node
        if subgraph:
            # First try to find line types connected to these nodes in the original graph
            adjacent_edge_types = []
            
            # Look at all adjacent edges of u and v in subgraph
            for node in [u, v]:
                for neighbor in subgraph.neighbors(node):
                    if subgraph.has_edge(node, neighbor):
                        edge_data = subgraph.edges[node, neighbor]
                        if "pandapower_type" in edge_data and isinstance(edge_data["pandapower_type"], dict):
                            adjacent_edge_types.append(edge_data["pandapower_type"])
            
            if adjacent_edge_types:
                # If we have adjacent edge types, pick the most common one
                selected_line_type = adjacent_edge_types[0]
                print(f"Using adjacent line type for {u}--{v}")
        
        # APPROACH 2: If no adjacent line types or no subgraph, use most common line type
        if not selected_line_type:
            # If multiple line types exist, use the most common one
            if len(existing_line_types) == 1:
                selected_line_type = existing_line_types[0]
            else:
                # Find the line type with highest capacity - usually a good choice
                # for reliability (this is a simple heuristic)
                selected_line_type = max(existing_line_types, 
                                        key=lambda x: x.get("max_i_ka", 0))
            
            print(f"Using common line type for {u}--{v}")
        
        # Ensure we don't have zero values for critical parameters
        for param in ["r_ohm_per_km", "x_ohm_per_km"]:
            if selected_line_type[param] <= 1e-10:
                # Use a small non-zero value to avoid division by zero
                selected_line_type[param] = 0.01
                print(f"WARNING: Fixed zero {param} for line {u}--{v}")
        
        create_line_and_switch(net, from_bus, to_bus, selected_line_type, line_sources,
                              line_type="interpolated", line_name=f"{u}--{v}: interpolated->selected")
    
    return net, line_sources

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

def create_line_and_switch(net, from_bus, to_bus, line_params, line_sources, line_type, line_name, is_switch=True):
    try:
        line_index = pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            r_ohm_per_km=line_params["r_ohm_per_km"],
            x_ohm_per_km=line_params["x_ohm_per_km"],
            c_nf_per_km=line_params["c_nf_per_km"],
            max_i_ka=line_params["max_i_ka"],
            type=line_params.get("type", "cs"),
            q_mm2=line_params["q_mm2"],
            alpha=line_params["alpha"],
            length_km =np.sqrt((net.bus_geodata.loc[from_bus, "x"] - net.bus_geodata.loc[to_bus, "x"])**2 + (net.bus_geodata.loc[from_bus, "y"] - net.bus_geodata.loc[to_bus, "y"])**2),
            name=line_name
        )
        line_sources[line_type] += 1
        pp.create_switch(net, bus=from_bus, element=line_index, et="l", closed=is_switch)
    except Exception as e:
        logger.error(f"Failed to create line {line_name}: {e}")
        line_sources["failed"] += 1


def scale_slack_buses(base_value, node_count, min_slack=1, max_slack=None):
    if max_slack is None:
        max_slack = min(int(node_count * 0.1), 20)

    min_slack = max(1, min_slack)
    max_slack = max(min_slack + 1, max_slack)
    
    small_graph = 50    
    large_graph = 500   
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
            print(f"Selected slack node {node} with metric {metric:.4f}")
        if len(selected_slack_nodes) >= num_slack:
            break
    if len(selected_slack_nodes) < num_slack:
        print(f"Could only select {len(selected_slack_nodes)} nodes with spacing constraint, filling remaining {num_slack - len(selected_slack_nodes)}")
        remaining_candidates = [node for node, _ in sorted_nodes if node not in selected_slack_nodes]
        additional_nodes = remaining_candidates[:min(num_slack - len(selected_slack_nodes), len(remaining_candidates))]
        
        for node in additional_nodes:
            selected_slack_nodes.append(node)
            print(f"Added additional slack node {node} with metric {metrics[node]}")
    return selected_slack_nodes


def create_pandapower_network(subgraph: nx.Graph, args: dict, dist_callable: dict) -> pp.pandapowerNet:
    net = pp.create_empty_network()

    base_num_slack = dist_callable["n_slack_busses"]()
    num_slack = scale_slack_buses(base_num_slack, len(subgraph.nodes))
    
    print(f"Graph has {len(subgraph.nodes)} nodes. Base slack buses: {base_num_slack}, Scaled slack buses: {num_slack}")
    
    vn_kv = find_operating_voltage(subgraph)
    print(f"Operating voltage: {vn_kv} kV")
    
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
        print(f"Creating slack generator for {slack_node}")
        bus_id = nx_to_pp_bus_map[slack_node]
        pp.create_gen(net, slack=True, bus=bus_id, vm_pu=1.02, p_mw=slack_power)
        net.bus.loc[bus_id, "name"] = f"{slack_node}_Slack"
        if slack_node in remaining_nodes:
            remaining_nodes.remove(slack_node)
    
    for node, data in subgraph.nodes(data=True):
        if node in remaining_nodes:
            bus_id = nx_to_pp_bus_map[node]
            net_load = data.get("net_load", 0)
            if net_load > 0:
                pp.create_load(net, bus=bus_id, p_mw=net_load, 
                               q_mvar=abs(net_load) * np.tan(np.arccos(0.9)))
                net.bus.loc[bus_id, "name"] = f"{node}_Load"
            elif net_load < 0:
                pp.create_gen(net, bus=bus_id, vm_pu=1.02, p_mw=abs(net_load))
                net.bus.loc[bus_id, "name"] = f"{node}_Gen"
            else:
                net.bus.loc[bus_id, "name"] = f"{node}_Neutral"
                
    failed_lines = []

    for u, v, edge_data in subgraph.edges(data=True):
        from_bus = nx_to_pp_bus_map[u]
        to_bus = nx_to_pp_bus_map[v]
        line_data = edge_data.get("pandapower_type", {})
        label = edge_data.get("label", "Unknown")
        is_switch = not edge_data.get("is_switch", False)
        try:
            create_line_and_switch(net, from_bus, to_bus, line_data, line_sources,
                                     line_type="original_data", line_name=f"{u}--{v}:{label}",
                                     is_switch=is_switch)
        except Exception as e:
            failed_lines.append((u, v))
    print("succesfull lines: ", (len(subgraph.edges) - len(failed_lines) )/ len(subgraph.edges), "%")

    if failed_lines:
        standard_cable_mapping = {
            'standard_cable_1': {"r_ohm_per_km": 0.124, "x_ohm_per_km": 0.08,
                                 "c_nf_per_km": 280, "max_i_ka": 0.32,
                                 "q_mm2": 150, "alpha": 0.00403, "type": "cs"},
            'standard_cable_2': {"r_ohm_per_km": 0.162, "x_ohm_per_km": 0.07,
                                 "c_nf_per_km": 310, "max_i_ka": 0.28,
                                 "q_mm2": 240, "alpha": 0.00403, "type": "cs"},
            'standard_cable_3': {"r_ohm_per_km": 0.193, "x_ohm_per_km": 0.06,
                                 "c_nf_per_km": 210, "max_i_ka": 0.25,
                                 "q_mm2": 95, "alpha": 0.00403, "type": "cs"}
        }
        chosen_cable_type = dist_callable["standard_cables"]()
        cable_data = standard_cable_mapping[chosen_cable_type]
        net, line_sources = interpolate_failed_lines(net, failed_lines, nx_to_pp_bus_map, cable_data, line_sources)
    
    logging.info(f"line_sources: {line_sources}")
    net.gen.loc[net.gen.slack, 'slack_weight'] = 1.0/len(net.gen[net.gen.slack])   


    r_ohm = net.line["r_ohm_per_km"]
    x_ohm = net.line["x_ohm_per_km"]
    
    # Handle empty series and make sure there are values before taking min
    if not r_ohm.empty and not x_ohm.empty:
        min_r = r_ohm.min()
        min_x = x_ohm.min()
        print("Minimum r and x values: ", min_r, min_x)
    else:
        print("No resistance or reactance values found")
    max_attempts = 5
    attempt = 0
    converged = False
    while attempt < max_attempts and not converged:
        try:
            pp.runpp(net, max_iteration=100, v_debug=True, run_control=True, initialization="dc",
                     calculate_voltage_angles=True)
            if net.converged:
                converged = True
                print("Successfully ran PP network after adjustments.")
                break
        except Exception as e:
            err_msg = str(e)
            print(f"Attempt {attempt+1}: Power flow failed with error: {err_msg}")
        for load_idx in net.load.index:
            net.load.at[load_idx, 'p_mw'] *= 0.50
        attempt += 1

    if not converged:
        # Identify candidate nodes (excluding existing slack) and add an additional slack generator.
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
            print(f"Added additional slack bus at node {new_slack}")
            try:
                pp.runpp(net, max_iteration=100, v_debug=True, run_control=True, initialization="dc",
                         calculate_voltage_angles=True)
                if net.converged:
                    converged = True
                    print("Successfully ran PP network after adding extra slack.")
            except Exception as e:
                print(f"Final attempt after adding extra slack failed: {e}")


    try:
        pp.runpp(net, max_iteration=100, v_debug=True, run_control=True, initialization="dc", 
                 #distributed_slack=True, 
                 calculate_voltage_angles=True)
        print("SUCCESFULLY RAN PP NETWORK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    except Exception as e:
        err_msg = str(e)
        print(f"Initial power flow failed: {err_msg}")
        net.converged = False
        match = re.search(r"Distributed slack contribution factors in island '([^']+)'", err_msg)
        if match:
            island_nodes_str = match.group(1).strip("[]")
            island_nodes = [s.strip() for s in island_nodes_str.split(',') if s.strip() != '']
            try:
                island_nodes = [int(x) for x in island_nodes]
            except Exception as e:
                print(f"Error converting island node strings to integers: {e}")
    #if island_nodes is not None: 
    #    plot_network(net, island_nodes)
    if args.show_pandapower_report:
        result = pp.diagnostic(net, detailed_report= True,warnings_only=False)
        print(result)

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
    #print(info)
    return net, info

def process_single_subgraph(subgraph, dfs, args, dist_callables, save_location):
    print("inside process_single_subgraph")
    graph_id = f"graph_{uuid.uuid4().hex}"
    suffixes = list(string.ascii_lowercase)
    succ = fail = 0

    original_subgraph = copy.deepcopy(subgraph)
    for sampled_idx in range(args.n_samples_per_graph):
        timeframes = sample_timeframes(args.n_loadcase_time_intervals, args.interval_duration_minutes)
        date_time = timeframes[0]
        #try:
        suffix = suffixes[sampled_idx % len(suffixes)]
        if args.modify_each_sample or sampled_idx == 0:
            logger.info(f"Modifying subgraph {graph_id} for sample {sampled_idx + 1}")
            modified_subgraph = modify_subgraph(original_subgraph, args, dist_callables)
        else:
            modified_subgraph = original_subgraph
        # select rondom date_Time  in year 2023
        
        
        # Sample timeframes
        #timeframes = sample_timeframes(kwargs["n_loadcase_time_intervals"], kwargs["interval_duration_minutes"])
                
        # Process each timeframe and save immediately
        #for idx,date_time in enumerate(timeframes):
        consumption, production, max_consumption,max_production = retrieve_standard_production(date_time, dfs[3])
        # Scale consumption and production based on average consumption per node
        consumption = 0.03 * consumption/max_consumption
        production =  0.03 * production/max_production

        for node in modified_subgraph.nodes:
            consumption_value = np.random.normal(consumption, consumption * args.consumption_std)
            production_value = np.random.normal(production, production * args.production_std)
            net_load = np.random.normal(consumption_value - production_value, abs(consumption_value - production_value)* args.net_load_std)                      
            modified_subgraph.nodes[node]["net_load"] = net_load

        electrified_network, info = create_pandapower_network(modified_subgraph, args, dist_callables)
        if electrified_network.converged:
            succ += 1
            name = f"{graph_id}_mod_{suffix}"
            save_single_graph(name, modified_subgraph, electrified_network, info, save_location)
            print(f"[Worker {os.getpid()}] ✅ Converged – about to save {graph_id}")
        else:
            fail += 1
            logger.info(f"Power flow failed for {graph_id}@{date_time}")
            print(f"[Worker {os.getpid()}]  Did *not* converge for {graph_id}@{date_time}")
                            
        #except Exception as e:
        #    logger.error(f"Error processing subgraph {subgraph_counter}: {e}")
        #    processed_counts["failed"] += 1
        #    continue
    print(f"[Worker {os.getpid()}] Finished processing subgraph {graph_id} with {succ} successes and {fail} failures.")
    return succ,fail

def get_n_workers():
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 1  # Fallback to 1 if cpu_count is not implemented    


def _worker(subgraph, dfs, args, dist_callables, save_location):
    """
    Process a single subgraph: returns (success_count, failure_count)
    """
    pid = os.getpid()
    random.seed()
    np.random.seed(int.from_bytes(os.urandom(4), 'little'))
    print(f"[Worker {pid}] , subgraph.nodes={len(subgraph.nodes)}")
    return process_single_subgraph(subgraph, dfs, args, dist_callables, save_location)


def transform_subgraphs(
    distributions: Dict[str, Any],
    dfs: Any,
    args: Any,
    logger: logging.Logger
) -> Dict[str, int]:
    """
    Lazily load subgraphs (or sample) and process in parallel.
    If not iterate_all, will loop until args.num_subgraphs datapoints created.
    """
    # initialize distributions
    dist_callables = initialize_distributions(distributions)

    # prepare save location
    save_dir = Path(args.save_dir or os.getcwd())
    total_required = len(raw_subgraphs) if args.iterate_all else args.num_subgraphs
    date_str = datetime.now().strftime("%d%m%Y")   
    mode_str = "all" if args.iterate_all else f"range-{args.target_busses}-{args.bus_range}"
    save_name =  f"{date_str}_{mode_str}_{total_required}"
    save_location = save_dir / save_name / "original"
    save_location.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {save_location}")

    
    nx_dir = save_location / "networkx_graphs"
    pp_dir = save_location/  "pandapower_networks"
    feat_dir = save_location/ "graph_features"
    for directory in [nx_dir, pp_dir, feat_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("start loading  subgraphs")

    # load all subgraphs once
    raw_subgraphs = load_subgraphs(args)
    if not raw_subgraphs:
        raise RuntimeError("No subgraphs found to process.")
    total_required = len(raw_subgraphs) if args.iterate_all else args.num_subgraphs

    processed = {"total": 0, "successful": 0, "failed": 0}
    print(" \n \n start transforming subgraphs")

    pbar = tqdm(total=total_required, desc="Processing subgraphs", unit="subgraph")

    # for idx, subgraph in enumerate(raw_subgraphs):
    #     _worker(subgraph, dfs, args, dist_callables, save_location, idx)


    # parallel execution
    with ProcessPoolExecutor(max_workers=get_n_workers()) as ex:
        while processed["total"] < total_required:
            # decide how many more to submit
            to_do = raw_subgraphs[: min(len(raw_subgraphs), total_required - processed["total"])]
            futures = [
                ex.submit(_worker, sg, dfs, args, dist_callables, save_location)
                for sg in to_do
            ]

            for fut in as_completed(futures):
                #try:
                suc, fl = fut.result()
                #except Exception as e:
                #    logger.exception("Worker crashed")
                #    suc, fl = 0, 1

                processed["successful"] += suc
                processed["failed"]     += fl
                processed["total"]      += (suc + fl)
                pbar.update(suc + fl)

    pbar.close()
    logger.info(f"Finished: {processed}")
    return processed

def extract_node_features(net, nx_graph, nx_to_pp_bus_map= None):
    if nx_to_pp_bus_map is None:
        nx_to_pp_bus_map = net["nx_to_pp_bus_map"]
    node_features = {}
    for node in nx_graph.nodes:
        pp_bus_idx = nx_to_pp_bus_map[node]
        node_features[node] = {
            "p": net.res_bus.p_mw.at[pp_bus_idx],
            "q": net.res_bus.q_mvar.at[pp_bus_idx],
            "v": net.res_bus.vm_pu.at[pp_bus_idx],
            "theta": net.res_bus.va_degree.at[pp_bus_idx]
        }
    #print(node_features)
    return node_features

def extract_edge_features(net, nx_graph, nx_to_pp_bus_map=None):
    if nx_to_pp_bus_map is None:
        nx_to_pp_bus_map = net["nx_to_pp_bus_map"]
    edge_features = {}
    for idx, (u, v) in enumerate(nx_graph.edges):
        pp_from_bus = nx_to_pp_bus_map[u]
        pp_to_bus = nx_to_pp_bus_map[v]
        matching_lines = net.line[(net.line.from_bus == pp_from_bus) & (net.line.to_bus == pp_to_bus)]
        if matching_lines.empty:
            continue
        line_idx = matching_lines.index[0]
        R = matching_lines.r_ohm_per_km.iloc[0]
        X = matching_lines.x_ohm_per_km.iloc[0]
        switch_status = int(net.switch[(net.switch.bus == pp_from_bus) & (net.switch.element == pp_to_bus)].closed.any())
        edge_features[(u, v)] = {
            "edge_idx": idx,
            "line_idx": line_idx,
            "R": R,
            "X": X,
            "switch_state": switch_status,
        }
    #print(edge_features)
    return edge_features

def print_feature_statistics(node_feats, edge_feats):
    """
    Print average values and statistics for node and edge features from feature dictionaries.
    
    Args:
        node_feats: Dictionary of node features
        edge_feats: Dictionary of edge features
    """
    # Collect node feature values
    node_features = {
        "p": [],
        "q": [],
        "v": [],
        "theta": []
    }
    
    for node_id, features in node_feats.items():
        for feature in node_features:
            if feature in features:
                value = features[feature]
                if not np.isnan(value):
                    node_features[feature].append(value)
    
    # Collect edge feature values
    edge_features = {
        "R": [],
        "X": [],
        "switch_state": []
    }
    
    for edge_id, features in edge_feats.items():
        for feature in edge_features:
            if feature in features:
                value = features[feature]
                if not np.isnan(value):
                    edge_features[feature].append(value)
    
    # Print node statistics
    print("\n---- Node Feature Statistics ----")
    for feature, values in node_features.items():
        if values:
            print(f"{feature:>6}: avg={np.mean(values):.4f}, min={np.min(values):.4f}, "
                  f"max={np.max(values):.4f}, std={np.std(values):.4f}, count={len(values)}")
        else:
            print(f"{feature:>6}: No valid values")
    
    # Print edge statistics
    print("\n---- Edge Feature Statistics ----")
    for feature, values in edge_features.items():
        if values:
            print(f"{feature:>12}: avg={np.mean(values):.4f}, min={np.min(values):.4f}, "
                  f"max={np.max(values):.4f}, std={np.std(values):.4f}, count={len(values)}")
        else:
            print(f"{feature:>12}: No valid values")
    
    # Print switch state counts
    switch_states = edge_features["switch_state"]
    if switch_states:
        switch_closed = sum(1 for s in switch_states if s > 0.5)
        switch_open = len(switch_states) - switch_closed
        print(f"\nSwitch states: {switch_closed} closed, {switch_open} open "
              f"({switch_closed/len(switch_states)*100:.1f}% closed)")
        
def build_clean_graph(nx_graph, node_feats, edge_feats):
    # Define the essential attributes we need
    essential_node_attrs = ["p", "q", "v", "theta"]
    essential_edge_attrs = ["R", "X", "switch_state"]
    
    clean_graph = nx.Graph()
    
    if hasattr(nx_graph, "name"):
        clean_graph.name = nx_graph.name
    
    # Relabel the graph nodes to consecutive integers
    node_mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    missing_node_attrs_count = 0
    
    for node in nx_graph.nodes():
        new_node_id = node_mapping[node]
        node_attrs = {"p": 0.0, "q": 0.0, "v": 0.0, "theta": 0.0}
        
        # Process each required attribute
        for attr in essential_node_attrs:
            # Try to get from node_feats first (preferred source)
            if node_feats and node in node_feats and attr in node_feats[node]:
                node_attrs[attr] = node_feats[node][attr]
            # If not in node_feats, try original graph
            elif attr in nx_graph.nodes[node]:
                node_attrs[attr] = nx_graph.nodes[node][attr]
            # If we got here, the attribute is truly missing (not in either source)
            # We'll use the default value from node_attrs initialization
        
        clean_graph.add_node(new_node_id, **node_attrs)

    missing_edge_attrs_count = 0
    
    for u, v in nx_graph.edges():
        new_u = node_mapping[u]
        new_v = node_mapping[v]
        
        edge_attrs = {
            "R": 0.00,
            "X": 0.00,
            "switch_state": 0.0
        }
        
        # Check both possible edge orientations in edge_feats
        edge = (u, v)
        reverse_edge = (v, u)
        edge_found = False
        
        # Process each required attribute
        for attr in essential_edge_attrs:
            # Try edge_feats first
            if edge_feats:
                if edge in edge_feats and attr in edge_feats[edge]:
                    edge_attrs[attr] = edge_feats[edge][attr]
                    edge_found = True
                elif reverse_edge in edge_feats and attr in edge_feats[reverse_edge]:
                    edge_attrs[attr] = edge_feats[reverse_edge][attr]
                    edge_found = True
                # If not in edge_feats, try the original graph
                elif attr in nx_graph[u][v]:
                    edge_attrs[attr] = nx_graph[u][v][attr]
                    edge_found = True
            # If edge_feats isn't available, try the original graph
            elif attr in nx_graph[u][v]:
                edge_attrs[attr] = nx_graph[u][v][attr]
                edge_found = True
                
        clean_graph.add_edge(new_u, new_v, **edge_attrs)
        
        if not edge_found:
            missing_edge_attrs_count += 1
    
    assert missing_node_attrs_count == 0, f"Missing node attributes: {missing_node_attrs_count}"
    assert missing_edge_attrs_count == 0, f"Missing edge attributes: {missing_edge_attrs_count}"
    return clean_graph

def save_single_graph(graph_name, nx_graph, pp_network, info, save_location):
    
    nx_dir = save_location / "networkx_graphs"
    pp_dir = save_location/  "pandapower_networks"
    feat_dir = save_location/ "graph_features"

    try:
        pp.runpp(pp_network, max_iteration=100, v_debug=False, run_control=True, initialization="dc", calculate_voltage_angles=True)
    except Exception as e:
        print(f"Power flow did not converge for {graph_name}: {e}")
    
    node_feats = extract_node_features(pp_network, nx_graph)
    edge_feats = extract_edge_features(pp_network, nx_graph)

    # Print feature statistics directly from the feature dictionaries
    print(f"\n===== Feature Statistics for {graph_name} =====")
    print_feature_statistics(node_feats, edge_feats)
    print("=" * 50)

    clean_graph = build_clean_graph(nx_graph, node_feats, edge_feats)

    print(len(clean_graph.nodes))
    print(clean_graph.nodes(data=True))
    
    features = {
    "node_features": {node: clean_graph.nodes[node] for node in clean_graph.nodes()},
    "edge_features": {(u, v): clean_graph.edges[u, v] for u, v in clean_graph.edges()},
    "info": info
    }

    nx_file = os.path.join(nx_dir, f"{graph_name}.pkl")
    with open(nx_file, "wb") as f:
        pkl.dump(clean_graph, f)

    pp_file = os.path.join(pp_dir, f"{graph_name}.json")
    with open(pp_file, "w") as f:
        json.dump(pp.to_json(pp_network), f)

    feat_file = os.path.join(feat_dir, f"{graph_name}.pkl")
    with open(feat_file, "wb") as f:
        pkl.dump(features, f)

    print(f"Saved graph {graph_name} to {save_location}")
    logger.info(f"[Worker {os.getpid()}] Saving graph {graph_name} to {nx_dir}")

