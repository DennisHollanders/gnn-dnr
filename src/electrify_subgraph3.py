import networkx as nx
import os
import matplotlib.pyplot as plt
import pandas as pd
import random 
import pickle as pkl
import pandapower as pp
import logging
import geopandas as gpd
from scipy.spatial import ConvexHull
from collections import defaultdict
import numpy as np
from typing import List, Dict, Tuple, Any
import copy
import string
from datetime import timedelta
import pprint
random.seed(0)
import ast 
from functools import partial
from pathlib import Path
import json 
from shapely.geometry import Point, LineString
import re
import glob
import math
import time
import simbench
import pandapower.networks as pn 
import concurrent.futures


from logger_setup import logger 
print(f"opened logger:{logger}")

# Ensure subgraph nodes have valid positions
def get_subgraph_centroid(subgraph):
    positions = [
        node_data["position"] for _, node_data in subgraph.nodes(data=True)
        if "position" in node_data and isinstance(node_data["position"], (list, tuple)) and len(node_data["position"]) == 2
    ]
    if not positions:
        raise ValueError("No valid positions found in the subgraph nodes.")
    avg_pos = np.mean(positions, axis=0)
    return Point(avg_pos)

# Find the nearest postcode6 for a given subgraph centroid
def find_closest_postcode6(cbs_pc6_gdf, subgraph):
    center = get_subgraph_centroid(subgraph)
    center_geo = gpd.GeoSeries([center], crs="EPSG:28992")  # Assume centroid is in EPSG:28992
    # Ensure CRS matches
    center_geo_projected = center_geo.to_crs(cbs_pc6_gdf.crs)
    # Calculate distances
    cbs_pc6_gdf["distance"] = cbs_pc6_gdf.geometry.distance(center_geo_projected.iloc[0])
    closest_row = cbs_pc6_gdf.loc[cbs_pc6_gdf["distance"].idxmin()]
    return closest_row["postcode6"]

# Match postcode6 to the corresponding buurtcode
def match_postcode6_to_buurt(postcode6, buurt_to_postcodes):
    match = buurt_to_postcodes.loc[buurt_to_postcodes["postcode6"] == postcode6, "buurtcode"]
    if not match.empty:
        return match.iloc[0]
    return None

# Find both the postcode6 and the corresponding buurtcode
def find_postcode6_and_buurt(cbs_pc6_gdf, buurt_to_postcodes, subgraph):
    closest_postcode6 = find_closest_postcode6(cbs_pc6_gdf, subgraph)
    matching_buurt = match_postcode6_to_buurt(closest_postcode6, buurt_to_postcodes)
    return closest_postcode6, matching_buurt

# Process multiple subgraphs
def find_postcode6s_and_buurts(subgraphs,dfs):
    cbs_pc6_gdf, buurt_to_postcodes = dfs[1], dfs[2]

    buurt_to_postcodes["postcode6"] = buurt_to_postcodes["postcode6"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    expanded_buurt_to_postcodes = buurt_to_postcodes.explode("postcode6").reset_index(drop=True)

    # Process each subgraph
    results = []
    for sg in subgraphs:
        try:
            postcode6, buurt = find_postcode6_and_buurt(cbs_pc6_gdf, expanded_buurt_to_postcodes, sg)
            print(f"Closest postcode6: {postcode6}, CBS Buurt: {buurt}")
            results.append({"postcode6": postcode6, "buurt": buurt})
        except ValueError as e:
            print(f"Error: {e}")
            results.append({"postcode6": None, "buurt": None})

    return pd.DataFrame(results)

from collections import defaultdict

def initialize_distributions(distributions, logger=None):
    distribution_samples = defaultdict(list)  # Track samples for each distribution

    def sample_from_distribution(dist_name: str, distribution: Dict[str, Any]) -> Any:
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

        distribution_samples[dist_name].append(sample)
        logger.info(f"Sampled {dist_name} -> {sample}")
        return sample

    dist_callables = {}
    for dist_name, dist_params in distributions.items():
        dist_callables[dist_name] = partial(sample_from_distribution, dist_name, dist_params)
    return dist_callables, distribution_samples


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

def modify_subgraph(subgraph, kwargs, dist_callable):
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
            subgraph_to_adapt, layers, convex_layers_to_use, max_distance, max_cycle_length, order_added, kwargs
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
    if kwargs["plot_added_edge"]:
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


def add_edge_via_convex_layers(subgraph_to_adapt, layers, layer_list, max_distance, max_cycle_length, order_added, kwargs):
    pos = nx.get_node_attributes(subgraph_to_adapt, 'position')
    selected_nodes = set(node for layer_index in layer_list if layer_index < len(layers) for node in layers[layer_index])
    selected_nodes = list(selected_nodes)

    possible_edges = []

    for i in range(len(selected_nodes)):
        for j in range(i + 1, len(selected_nodes)):
            u, v = selected_nodes[i], selected_nodes[j]
            if subgraph_to_adapt.has_edge(u, v):
                continue  
            distance = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
            if distance < kwargs.get('min_distance_threshold', 1.0):
                continue  
            edge = (u, v)
            score = check_and_add_edge(edge, subgraph_to_adapt, pos, max_distance, max_cycle_length, kwargs['weight_factor'])
            if score is not None:
                possible_edges.append((edge, score))

    possible_edges.sort(key=lambda x: x[1], reverse=True)

    if not possible_edges:
        print("No valid edges to add.")
        return subgraph_to_adapt, False

    if kwargs['deterministic']:
        best_edge = possible_edges[0][0]
    else:
        top_edges = possible_edges[:kwargs['top_x']]
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


# def interpolate_failed_lines(net, failed_lines, nx_to_pp_bus_map, random_cable_data, line_sources):
#     if len(net.line) > 0:
#         print("Interpolating failed lines.")
#         avg_params = {
#             "r_ohm_per_km": net.line["r_ohm_per_km"].mean(),
#             "x_ohm_per_km": net.line["x_ohm_per_km"].mean(),
#             "c_nf_per_km": net.line["c_nf_per_km"].mean(),
#             "max_i_ka": net.line["max_i_ka"].mean(),
#             "q_mm2": net.line["q_mm2"].mean(),
#             "alpha": net.line["alpha"].mean()
#         }
#         for u, v in failed_lines:
#             from_bus = nx_to_pp_bus_map[u]
#             to_bus = nx_to_pp_bus_map[v]
#             create_line_and_switch(net, from_bus, to_bus, avg_params, line_sources,
#                                      line_type="interpolated", line_name=f"{u}--{v}: interpolated->average")
#     else:
#         print("Creating random lines for failed lines.")	
#         for u, v in failed_lines:
#             from_bus = nx_to_pp_bus_map[u]
#             to_bus = nx_to_pp_bus_map[v]
#             create_line_and_switch(net, from_bus, to_bus, random_cable_data, line_sources,
#                                      line_type="standard_cable", line_name=f"{u}--{v}: interpolated->random:{random_cable_data}")
#     return net, line_sources
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
        print(f"Failed to create line {line_name}: {e}")
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
                # Nodes are in different components, so they're far apart
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


def create_pandapower_network(subgraph: nx.Graph, kwargs: dict, dist_callable: dict) -> pp.pandapowerNet:
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
    if kwargs["show_pandapower_report"]:
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


def electrify_graphs(subgraphs, dfs, kwargs, dist_callable, save_location=None):
    consumption_df = dfs[0]
    postal_code_list = find_postcode6s_and_buurts(subgraphs, dfs)
    clustered_subgraphs = {}  # Key: CBS buurt, Value: list of subgraphs
    for subgraph, row in zip(subgraphs, postal_code_list.itertuples(index=False)):
        cbs_buurt = row.buurt
        if cbs_buurt not in clustered_subgraphs:
            clustered_subgraphs[cbs_buurt] = []
        clustered_subgraphs[cbs_buurt].append(subgraph)

    suffixes = list(string.ascii_lowercase)
    subgraph_counter = 0

    # Dictionary to track counts for reporting purposes
    processed_counts = {"total": 0, "successful": 0, "failed": 0}

    for cbs_buurt, subgraph_list in clustered_subgraphs.items():
        total_nodes = sum(len(subgraph.nodes) for subgraph in subgraph_list)
        try:
            neighborhood_df = consumption_df[consumption_df["CBS Buurtcode"] == cbs_buurt]
            sja_value = neighborhood_df['kleinverbruik_SJA_GEMIDDELD'].values[0]
            percentage_gen = (neighborhood_df['opwek_klein_AANTAL_AANSLUITINGEN_MET_OPWEKINSTALLATIE'].values[0] + 1) / \
                             (neighborhood_df['opwek_klein_AANTAL_AANSLUITINGEN_IN_CBS_BUURT'].values[0] + 1)
            opwek_groot = neighborhood_df[[
                "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_WIND_IN_AANLEG",
                "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_WIND_IN_BEDRIJF",
                "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_ZON_IN_AANLEG",
                "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_ZON_IN_BEDRIJF"
            ]].sum(axis=1).values[0]
            ja_nb = sja_value * neighborhood_df["kleinverbruik_AANSLUITINGEN_AANTAL"]
            ja_mw = ja_nb / (8760 * 4 * 1000)  # Convert from kWh to MW
            opwek_groot = opwek_groot / (8760 * 4 * 1000)  # Convert from kWh to MW
            avg_consumption_per_node = ja_mw / (total_nodes + 1)
        except IndexError:
            logging.warning(f"Data for CBS buurt {cbs_buurt} not found in consumption_df.")
            continue

        clustered_subgraphs[cbs_buurt] = {
            "subgraphs": subgraph_list,
            "info": {
                "total_nodes": total_nodes,
                "percentage_gen": percentage_gen,
                "opwek_groot": opwek_groot,
                "avg_consumption_per_node": avg_consumption_per_node,
            },
        }

        for subgraph in subgraph_list:
            graph_id = f"graph_{subgraph_counter}"
            processed_counts["total"] += 1

            original_subgraph = copy.deepcopy(subgraph)
            for sampled_idx in range(kwargs['n_samples_per_graph']):
                try:
                    suffix = suffixes[sampled_idx % len(suffixes)]
                    if kwargs['modify_subgraph_each_sample'] or sampled_idx == 0:
                        logger.info(f"Modifying subgraph {subgraph_counter} for sample {sampled_idx + 1}")
                        modified_subgraph = modify_subgraph(original_subgraph, kwargs, dist_callable)
                    else:
                        modified_subgraph = original_subgraph
                        
                    # Sample timeframes
                    timeframes = sample_timeframes(kwargs["n_loadcase_time_intervals"], kwargs["interval_duration_minutes"])
                    
                    # Process each timeframe and save immediately
                    for idx,date_time in enumerate(timeframes):
                        consumption, production, max_consumption,max_production = retrieve_standard_production(date_time, dfs[3])
                        # Scale consumption and production based on average consumption per node
                        consumption = avg_consumption_per_node * consumption/max_consumption
                        production =  opwek_groot * production/max_production

                        for node in modified_subgraph.nodes:
                            consumption_value = np.random.normal(consumption, consumption * kwargs["consumption_std"])
                            production_value = np.random.normal(production, production * kwargs["production_std"])
                            net_load = np.random.normal(consumption_value - production_value, abs(consumption_value - production_value)* kwargs["net_load_std"])                      
                            modified_subgraph.nodes[node]["net_load"] = net_load
                        electrified_network, info = create_pandapower_network(modified_subgraph, kwargs, dist_callable)
                        if electrified_network.converged:
                            processed_counts["successful"] += 1
                            if kwargs["save"] and save_location:
                                graph_name = f"{graph_id}_modification_{suffix}_{idx}"
                                save_single_graph(
                                    graph_name, 
                                    modified_subgraph, 
                                    electrified_network, 
                                    info, 
                                    save_location
                                )
                        else:
                            processed_counts["failed"] += 1
                            logger.info(f"Power flow failed for subgraph {subgraph_counter} at {date_time}")
                            
                except Exception as e:
                    logger.error(f"Error processing subgraph {subgraph_counter}: {e}")
                    processed_counts["failed"] += 1
                    continue
                
            subgraph_counter += 1
            
    logger.info(f"Processing summary: Total: {processed_counts['total']}, Successful: {processed_counts['successful']}, Failed: {processed_counts['failed']}")
    return processed_counts  

def transform_subgraphs(subgraphs: List[nx.Graph],
                        distributions: Dict[str, Any],
                        dfs: Any, kwargs: Dict[str, Any], logger) -> Tuple[Dict[str, int], List[nx.Graph]]:
    # Initialize distributions and collect samples
    dist_callables, distribution_samples = initialize_distributions(distributions, logger)
    
    # Make sure save_location is passed to electrify_graphs
    save_location = kwargs.get("save_location", None)
    if kwargs["save"] and not save_location:
        logger.warning("Save option is enabled but no save_location provided. Using default './saved_graphs'")
        save_location = "./saved_graphs"
        
    if kwargs['is_iterate']:
        logger.info("Iterating over all subgraphs.")
        print("iterating over all subgraphs")
        processing_stats = electrify_graphs(subgraphs, dfs, kwargs, dist_callables, save_location)
    else:
        logger.info("Sampling subgraphs.")
        print("sampling subgraphs")
        n_busses_target = dist_callables["n_busses"]()
        n_busses_range = (n_busses_target - kwargs['range'], n_busses_target + kwargs['range'])
        print("n_busses_range", n_busses_range)
        logger.info(f"Target number of busses: {n_busses_target}, range: {n_busses_range}")
        filtered_subgraphs = [g for g in subgraphs if n_busses_range[0] <= len(g.nodes) <= n_busses_range[1]]

        if not filtered_subgraphs:
            logger.info(f"No subgraphs found in range {n_busses_range}. Sampling randomly.")
            graphs_to_electrify = random.choices(subgraphs, k=kwargs['amount_of_subgraphs'])
        else:
            graphs_to_electrify = random.choices(filtered_subgraphs, k=kwargs['amount_of_subgraphs'])
        
        processing_stats = electrify_graphs(graphs_to_electrify, dfs, kwargs, dist_callables, save_location)

    if kwargs['plot_distributions']:
        plot_distributions(distributions, distribution_samples)

    logger.info("Transformation process completed.")
    logger.info(f"Processing summary: {processing_stats}")

    # Ensure log handlers are closed after the process
    if logger:
        for dist_name, samples in distribution_samples.items():
            logger.info(f"Samples for {dist_name}: {samples}")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    return processing_stats

def extract_node_features(net, nx_graph):
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

def extract_edge_features(net, nx_graph):
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
            # Try edge_feats first (in both orientations)
            if edge_feats:
                if edge in edge_feats and attr in edge_feats[edge]:
                    edge_attrs[attr] = edge_feats[edge][attr]
                    edge_found = True
                elif reverse_edge in edge_feats and attr in edge_feats[reverse_edge]:
                    edge_attrs[attr] = edge_feats[reverse_edge][attr]
                    edge_found = True
                # If not in edge_feats, try original graph
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
    nx_dir = os.path.join(save_location, "networkx_graphs")
    pp_dir = os.path.join(save_location, "pandapower_networks")
    feat_dir = os.path.join(save_location, "graph_features")

    for directory in [nx_dir, pp_dir, feat_dir]:
        os.makedirs(directory, exist_ok=True)

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


#==================================================================================================
#                               code for validation and test sets
#==================================================================================================



def has_switches(net):
    return hasattr(net, 'switch') and not net.switch.empty and len(net.switch) > 0

def load_network(network_id):
    try:
        if network_id.startswith('simbench_'):
            code = network_id[9:]  
            net = simbench.get_simbench_net(code)
        elif network_id.startswith('pp_'):
            case = network_id[3:]  #
            if case == "caseIEEE30":
                net = pn.case_IEEE30() if hasattr(pn, "case_IEEE30") else pn.case30()
            else:
                net = getattr(pn, case)()
        else:
            raise ValueError(f"Unknown network type: {network_id}")
        return net
    except Exception as e:
        print(f"Error loading network {network_id}: {e}")
        return None

def check_network_suitability(network_id, bus_range=(25,50), require_switches=True):
    net = load_network(network_id)
    
    if net is None:
        return False, None

    if not (bus_range[0] <= len(net.bus) <= bus_range[1]):
        return False, None

    if require_switches and not has_switches(net):
        return False, None
    
    return True, net

def get_candidate_networks(bus_range=(25,50), require_switches=True, max_workers=4):
    """
    Get all candidate networks that meet the specified criteria
    """
    start_time = time.time()
    candidate_networks = {}
    networks_without_switches = []
    
    # Get Simbench network codes
    list_of_codes = simbench.collect_all_simbench_codes(mv_level="MV")
    mv_codes = [code for code in list_of_codes if "MV" in code]
    
    # Filter Simbench codes if require_switches
    if require_switches:
        mv_sw_codes = [code for code in mv_codes if "no_sw" not in code]
    else:
        mv_sw_codes = mv_codes
    
    # Get PandaPower network names
    standard_cases = [
        "case4gs", "case5", "case6ww", "case9", "case14", "case30", 
        "caseIEEE30", "case33bw", "case39", "case57", "case89pegase", 
        "case118", "case145"
    ]
    
    # Create network IDs for all potential candidates
    simbench_ids = [f"simbench_{code}" for code in mv_sw_codes]
    pp_ids = [f"pp_{case}" for case in standard_cases]
    all_network_ids = simbench_ids + pp_ids
    
    print(f"Checking {len(all_network_ids)} potential networks...")
    
    # Use parallel processing to check network suitability
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(check_network_suitability, 
                           network_id, 
                           bus_range, 
                           require_switches): network_id 
            for network_id in all_network_ids
        }
        
        for future in concurrent.futures.as_completed(future_to_id):
            network_id = future_to_id[future]
            try:
                suitable, net = future.result()
                if suitable and net is not None:
                    bus_count = len(net.bus)
                    switch_count = len(net.switch) if hasattr(net, 'switch') else 0
                    candidate_networks[network_id] = net
                    print(f"Added network {network_id} with {bus_count} buses and {switch_count} switches")
                elif require_switches and net is not None and not has_switches(net):
                    networks_without_switches.append(network_id)
            except Exception as e:
                print(f"Error processing {network_id}: {e}")
    
    print(f"\nFound {len(candidate_networks)} valid networks matching criteria in {time.time() - start_time:.2f}s")
    if require_switches and networks_without_switches:
        print(f"Skipped {len(networks_without_switches)} networks without switches")
    
    return candidate_networks

def create_network_case(network_id, net, case_type, case_idx, load_variation_range=(0.5, 1.51)):
    """Create a single network case with variations"""
    try:
        # Apply load variations (create a deep copy to avoid modifying the original)
        net_case = copy.deepcopy(net)
        
        # Apply random load variations
        for idx in net_case.load.index:
            factor = np.random.uniform(load_variation_range[0], load_variation_range[1])
            net_case.load.at[idx, "p_mw"] *= factor
            net_case.load.at[idx, "q_mvar"] *= factor
        
        # Create case name
        case_name = f"{network_id}_{case_type}_{case_idx}"
        
        # Create NetworkX graph respecting switches
        nx_graph = top.create_nxgraph(net_case, respect_switches=True)
        
        switch_count = len(net_case.switch) if hasattr(net_case, 'switch') else 0
        print(f"Added {case_type} case {case_name} with {switch_count} switches")
        
        return case_name, {"network": net_case, "nx_graph": nx_graph}
    except Exception as e:
        print(f"Failed to create case for {network_id}: {e}")
        return None, None

def generate_combined_dataset(bus_range=(25,50), test_total_cases=100, val_total_cases=50, 
                              load_variation_range=(0.5,1.50), random_seed=1,
                              require_switches=True, max_workers=4):    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    print("Step 1: Finding suitable networks...")
    candidate_networks = get_candidate_networks(
        bus_range=bus_range, 
        require_switches=require_switches,
        max_workers=max_workers
    )
    
    if len(candidate_networks) == 0:
        raise ValueError("No networks found matching the criteria. Cannot generate dataset.")
    
    net_keys = list(candidate_networks.keys())
    random.shuffle(net_keys)
   
    test_dataset = {}
    val_dataset = {}
    
    print("\n Step 2: Generating test cases...")
    test_count = 0
    test_index = 0
    
    while test_count < test_total_cases and test_index < len(net_keys):
        key = net_keys[test_index]
        base_net = candidate_networks[key]
        
        # Calculate how many cases we need from this network
        remaining_test_cases = test_total_cases - test_count
        
        for i in range(remaining_test_cases):
            case_name, case_data = create_network_case(
                key, base_net, "test", i, load_variation_range
            )
            
            if case_name is not None and case_data is not None:
                test_dataset[case_name] = case_data
                test_count += 1
        
        test_index += 1
    
    if test_count < test_total_cases:
        print(f"Warning: Could only generate {test_count}/{test_total_cases} test cases with original strategy.")
        print("Applying additional load variations to existing networks...")
        
        existing_keys = list(set([name.split("_test_")[0] for name in test_dataset.keys()]))
        
        additional_cases_needed = test_total_cases - test_count
        cases_per_net = math.ceil(additional_cases_needed / max(1, len(existing_keys)))
        
        for key in existing_keys:
            if test_count >= test_total_cases:
                break
                
            if key in candidate_networks:
                base_net = candidate_networks[key]
                
                existing_variations = [int(name.split("_test_")[1]) for name in test_dataset.keys() 
                                     if name.startswith(f"{key}_test_")]
                start_idx = max(existing_variations) + 1 if existing_variations else 0
                
                for i in range(start_idx, start_idx + cases_per_net):
                    if test_count >= test_total_cases:
                        break
                    
                    # Use slightly different load variation range for diversity
                    var_range = (
                        load_variation_range[0] * 0.9,  # More extreme lower bound
                        load_variation_range[1] * 1.1   # More extreme upper bound
                    )
                    
                    case_name, case_data = create_network_case(
                        key, base_net, "test", i, var_range
                    )
                    
                    if case_name is not None and case_data is not None:
                        test_dataset[case_name] = case_data
                        test_count += 1
    
    print("\nStep 3: Generating validation cases...")
    val_count = 0
    val_index = 0
    
    # Exclude keys already used in test set to avoid overlap
    val_candidate_keys = [key for key in net_keys if key not in 
                         [name.split("_test_")[0] for name in test_dataset.keys()]]
    
    while val_count < val_total_cases and val_index < len(val_candidate_keys):
        key = val_candidate_keys[val_index]
        
        # Skip if already used for test set
        if any(case_name.startswith(key) for case_name in test_dataset.keys()):
            val_index += 1
            continue
            
        base_net = candidate_networks[key]
        
        # Calculate how many validation cases we need from this network
        remaining_val_cases = val_total_cases - val_count
        
        for i in range(remaining_val_cases):
            case_name, case_data = create_network_case(
                key, base_net, "val", i, load_variation_range
            )
            
            if case_name is not None and case_data is not None:
                val_dataset[case_name] = case_data
                val_count += 1
        
        val_index += 1
    
    if val_count < val_total_cases:
        print(f"Warning: Could only generate {val_count}/{val_total_cases} validation cases with original strategy.")
        
        # Use whatever networks are left, or if necessary, reuse test networks with different variations
        remaining_keys = [key for key in net_keys if key not in 
                         [name.split("_val_")[0] for name in val_dataset.keys()]]
        
        if not remaining_keys:
            # If no more unused networks, reuse some test networks with different variations
            remaining_keys = list(set([name.split("_test_")[0] for name in test_dataset.keys()]))
            print("Using test networks with different variations for validation...")
        
        additional_cases_needed = val_total_cases - val_count
        cases_per_net = math.ceil(additional_cases_needed / max(1, len(remaining_keys)))
        
        for key in remaining_keys:
            if val_count >= val_total_cases:
                break
                
            base_net = candidate_networks[key]
            
            existing_variations = [int(name.split("_val_")[1]) for name in val_dataset.keys() 
                                 if name.startswith(f"{key}_val_")]
            start_idx = max(existing_variations) + 1 if existing_variations else 0
            
            for i in range(start_idx, start_idx + cases_per_net):
                if val_count >= val_total_cases:
                    break
                
                case_name, case_data = create_network_case(
                    key, base_net, "val", i, load_variation_range
                )
                
                if case_name is not None and case_data is not None:
                    val_dataset[case_name] = case_data
                    val_count += 1
    
    # check if sets meet targets
    if len(test_dataset) < test_total_cases:
        print(f"WARNING: Could only generate {len(test_dataset)} test cases out of {test_total_cases} requested.")
    
    if len(val_dataset) < val_total_cases:
        print(f"WARNING: Could only generate {len(val_dataset)} validation cases out of {val_total_cases} requested.")
    
    # Print statistics on the generated datasets
    print("\n--- Dataset Statistics ---")
    test_networks = set([k.split("_test_")[0] for k in test_dataset.keys()])
    val_networks = set([k.split("_val_")[0] for k in val_dataset.keys()])
    
    print(f"Test dataset: {len(test_dataset)} cases from {len(test_networks)} unique networks")
    print(f"Validation dataset: {len(val_dataset)} cases from {len(val_networks)} unique networks")
    
    # Count switches in each dataset
    test_switch_counts = [len(data["network"].switch) if hasattr(data["network"], "switch") else 0 
                         for data in test_dataset.values()]
    val_switch_counts = [len(data["network"].switch) if hasattr(data["network"], "switch") else 0 
                        for data in val_dataset.values()]
    
    print(f"Test dataset switch statistics: Min={min(test_switch_counts) if test_switch_counts else 0}, "
          f"Max={max(test_switch_counts) if test_switch_counts else 0}, "
          f"Avg={sum(test_switch_counts)/len(test_switch_counts) if test_switch_counts else 0:.2f}")
    print(f"Validation dataset switch statistics: Min={min(val_switch_counts) if val_switch_counts else 0}, "
          f"Max={max(val_switch_counts) if val_switch_counts else 0}, "
          f"Avg={sum(val_switch_counts)/len(val_switch_counts) if val_switch_counts else 0:.2f}")
    
    return test_dataset, val_dataset

def save_combined_data(dataset, set_name, base_dir):
    nx_dir = os.path.join(base_dir, set_name, "networkx_graphs")
    pp_dir = os.path.join(base_dir, set_name, "pandapower_networks")
    feat_dir = os.path.join(base_dir, set_name, "graph_features")
    
    for directory in [nx_dir, pp_dir, feat_dir]:
        os.makedirs(directory, exist_ok=True)
    
    for case_name, data in dataset.items():
        net = data["network"]
        nx_graph = data["nx_graph"]

        try:
            pp.runpp(net, max_iteration=100, v_debug=False, run_control=True, 
                    initialization="dc", calculate_voltage_angles=True)
        except Exception as e:
            print(f"Power flow did not converge for {case_name}: {e}")
            
        node_feats = extract_node_features(net, nx_graph) 
        edge_feats = extract_edge_features(net, nx_graph) 
        features = {"node_features": node_feats, "edge_features": edge_feats}
            
        nx_file = os.path.join(nx_dir, f"{case_name}.pkl")
        with open(nx_file, "wb") as f:
            pkl.dump(nx_graph, f)
            
        pp_file = os.path.join(pp_dir, f"{case_name}.json")
        with open(pp_file, "w") as f:
            json.dump(pp.to_json(net), f)

        feat_file = os.path.join(feat_dir, f"{case_name}.pkl")
        with open(feat_file, "wb") as f:
            pkl.dump(features, f)
    
    print(f"Saved {len(dataset)} {set_name} cases individually to {base_dir}")

def save_dataset(test_dataset, val_dataset, base_path, bus_range, test_cases, val_cases):
    now = datetime.now()
    day = now.day
    month = now.month
    year = now.year

    bus_range_str = f"{bus_range[0]}-{bus_range[1]}"
    
    base_name = f"test_val_real__range-{bus_range_str}_nTest-{test_cases}_nVal-{val_cases}_{day}{month}{year}"

    search_pattern = f"{base_name}_*"

    existing_dirs = glob.glob(os.path.join(base_path, search_pattern))
 
    if not existing_dirs:
        sequence_num = 1
    else:
        seq_nums = []
        for dir_path in existing_dirs:
            try:
                seq_num = int(os.path.basename(dir_path).split('_')[-1])
                seq_nums.append(seq_num)
            except (ValueError, IndexError):
                continue
        
        sequence_num = max(seq_nums) + 1 if seq_nums else 1
    

    dataset_dir = f"{base_name}_{sequence_num}"
    save_location = os.path.join(base_path, dataset_dir)
    
    print(f"Creating dataset at: {save_location}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    save_combined_data(test_dataset, "test", save_location)
    save_combined_data(val_dataset, "validation", save_location)
    
    print(f"Dataset saved successfully at {save_location}")
    return save_location