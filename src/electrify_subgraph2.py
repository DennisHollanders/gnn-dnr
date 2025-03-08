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
    print("shouldplot_added_edge", kwargs["plot_added_edge"])
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
        return longest_paths[0] + 1  # Single longest path plus one edge
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


def interpolate_failed_lines(net, failed_lines, nx_to_pp_bus_map, random_cable_data, line_sources):
    if len(net.line) > 0:
        avg_params = {
            "r_ohm_per_km": net.line["r_ohm_per_km"].mean(),
            "x_ohm_per_km": net.line["x_ohm_per_km"].mean(),
            "c_nf_per_km": net.line["c_nf_per_km"].mean(),
            "max_i_ka": net.line["max_i_ka"].mean(),
            "q_mm2": net.line["q_mm2"].mean(),
            "alpha": net.line["alpha"].mean()
        }
        for u, v in failed_lines:
            from_bus = nx_to_pp_bus_map[u]
            to_bus = nx_to_pp_bus_map[v]
            create_line_and_switch(net, from_bus, to_bus, avg_params, line_sources,
                                     line_type="interpolated", line_name=f"{u}--{v}: interpolated->average")
    else:
        for u, v in failed_lines:
            from_bus = nx_to_pp_bus_map[u]
            to_bus = nx_to_pp_bus_map[v]
            create_line_and_switch(net, from_bus, to_bus, random_cable_data, line_sources,
                                     line_type="standard_cable", line_name=f"{u}--{v}: interpolated->random:{random_cable_data}")
    return net, line_sources

def find_operating_voltage(subgraph):
    for _, _, data in subgraph.edges(data=True):
        if 'operatingvoltage' in data and not pd.isna(data['operatingvoltage']):
            return data['operatingvoltage'] / 1000
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

def create_pandapower_network(subgraph: nx.Graph, kwargs: dict, dist_callable: dict) -> pp.pandapowerNet:
    net = pp.create_empty_network()
    num_slack = 4 # dist_callable["n_slack"]()
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
    
    metrics = {node: compute_slack_metric(subgraph, node) for node in candidate_nodes}
    max_metric = max(metrics.values())
    top_candidates = [node for node, m in metrics.items() if m == max_metric]
    selected_slack_nodes = random.sample(top_candidates, k=min(num_slack, len(top_candidates)))

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
    net.sn_mva = 10.0
 
    island_nodes = None

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
    print(info)
    return net, info


def electrify_graphs(subgraphs, dfs, kwargs, dist_callable):
    consumption_df = dfs[0]
    postal_code_list = find_postcode6s_and_buurts(subgraphs, dfs)
    clustered_subgraphs = {}  # Key: CBS buurt, Value: list of subgraphs
    for subgraph, row in zip(subgraphs, postal_code_list.itertuples(index=False)):
        cbs_buurt = row.buurt
        if cbs_buurt not in clustered_subgraphs:
            clustered_subgraphs[cbs_buurt] = []
        clustered_subgraphs[cbs_buurt].append(subgraph)

    suffixes = list(string.ascii_lowercase)
    electrified_subgraphs = {}
    subgraph_counter = 0

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
            electrified_subgraphs[f"graph_{subgraph_counter}"] = {
                "original_graph": subgraph,
                "modified_graphs": {}
            }
            dict_of_modified_subgraphs = {}
            original_subgraph = copy.deepcopy(subgraph)
            for sampled_idx in range(kwargs['n_samples_per_graph']):
                suffix = suffixes[sampled_idx % len(suffixes)]
                if kwargs['modify_subgraph_each_sample'] or sampled_idx == 0:
                    logger.info(f"Modifying subgraph {original_subgraph} for sample {sampled_idx + 1}")
                    modified_subgraph = modify_subgraph(original_subgraph, kwargs, dist_callable)
                else:
                    modified_subgraph = original_subgraph

                timeframes = sample_timeframes(kwargs["n_loadcase_time_intervals"], kwargs["interval_duration_minutes"])
                load_case_of_modified_subgraph = {}

                for date_time in timeframes:
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
                        load_case_of_modified_subgraph[str(date_time)] = {"network": electrified_network, "info": info}
                    else:
                        logger.info(f"Power flow failed for subgraph {original_subgraph} at {date_time}")
                        load_case_of_modified_subgraph[str(date_time)] = {"network": None, "info": info}

                dict_of_modified_subgraphs[suffix] = {
                    "subgraph": modified_subgraph,
                    "loadcases": load_case_of_modified_subgraph,
                }

            electrified_subgraphs[f"graph_{subgraph_counter}"]["modified_graphs"] = dict_of_modified_subgraphs
            subgraph_counter += 1

    formatted_subgraphs = pprint.pformat(electrified_subgraphs, indent=4, width=120)
    logger.info(f"Electrified subgraphs:\n{formatted_subgraphs}")
    return electrified_subgraphs

def transform_subgraphs(subgraphs: List[nx.Graph],
                        distributions: Dict[str, Any],
                        dfs: Any, kwargs: Dict[str, Any], logger) -> Tuple[List[Any], List[nx.Graph]]:
    # Initialize distributions and collect samples
    dist_callables, distribution_samples = initialize_distributions(distributions, logger)

    print("start")

    if kwargs['is_iterate']:
        logger.info("Iterating over all subgraphs.")
        print("iterating over all subgraphs")
        electrified_graphs = electrify_graphs(subgraphs, dfs, kwargs, dist_callables)
        print("got here")

    else:
        logger.info("Sampling subgraphs.")
        print("sampling subgraphs")
        n_busses_target = dist_callables["n_busses"]()
        n_busses_range = (n_busses_target - kwargs['range'], n_busses_target + kwargs['range'])
        filtered_subgraphs = [g for g in subgraphs if n_busses_range[0] <= len(g.nodes) <= n_busses_range[1]]

        if not filtered_subgraphs:
            logger.info(f"No subgraphs found in range {n_busses_range}. Sampling randomly.")
            graphs_to_electrify = random.sample(subgraphs, kwargs['amount_of_subgraphs'])
        else:
            graphs_to_electrify = random.choices(filtered_subgraphs, k=kwargs['amount_of_subgraphs'])
        
        electrified_graphs = electrify_graphs(graphs_to_electrify, dfs, kwargs, dist_callables)

    if kwargs['plot_subgraphs']:
        plot_sampled_subgraphs(
            [g for g in electrified_graphs.values() if g is not None][:kwargs['amount_to_plot']], 
            [g for g in electrified_graphs.values() if g is not None][:kwargs['amount_to_plot']]
        )
    if kwargs['plot_distributions']:
        plot_distributions(distributions, graph_parameters)

    # kwarg saving
    if kwargs["save"]:
        save_graph_data(electrified_graphs, kwargs, distributions, distribution_samples)

    logger.info("Transformation process completed.")

    # Ensure log handlers are closed after the process
    if logger:
        for dist_name, samples in distribution_samples.items():
            logger.info(f"Samples for {dist_name}: {samples}")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    return electrified_graphs

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
    return edge_features

def save_graph_data(electrified_graphs, kwargs, distributions, distribution_samples):
    """
    Save networkx graphs as a dictionary in a Pickle file,
    Pandapower networks as JSON, and features in a separate Pickle file.
    """
    save_location = kwargs["save_location"]
    os.makedirs(save_location, exist_ok=True)
    
    print("Starting the saving process")

    nx_graphs = {}
    pp_networks = {}
    features = {}

    for graph_id, graph_data in electrified_graphs.items():
        if graph_data is None:
            print(f"Skipping graph {graph_id} as it did not converge.")
            continue

        original_graph = graph_data["original_graph"]
        modified_graphs = graph_data.get("modified_graphs", {})

        for suffix, modified_graph_data in modified_graphs.items():
            modified_subgraph = modified_graph_data["subgraph"]
            loadcases = modified_graph_data["loadcases"]

            for timestamp, loadcase_data in loadcases.items():
                if loadcase_data is None:
                    print(f"Skipping {graph_id} - {suffix} - {timestamp} due to non-converged network.")
                    continue

                graph_name = f"{graph_id}_modification_{suffix}_{timestamp}"

                # Store NetworkX graph
                nx_graphs[graph_name] = modified_subgraph

                if loadcase_data["network"]:
                    pp_networks[graph_name] = pp.to_json(loadcase_data["network"])

                # Extract node & edge features
                node_features = extract_node_features(loadcase_data["network"], modified_subgraph) if loadcase_data["network"] else None
                edge_features = extract_edge_features(loadcase_data["network"], modified_subgraph) if loadcase_data["network"] else None

                features[graph_name] = {"node_features": node_features, "edge_features": edge_features}

    # Save NetworkX graphs as a Pickle file
    with open(f"{save_location}/networkx_graphs.pkl", "wb") as f:
        pkl.dump(nx_graphs, f)
    print("Saved NetworkX graphs.")

    # Save Pandapower networks as a JSON file
    with open(f"{save_location}/pandapower_networks.json", "w") as f:
        f.write(pp.to_json(pp_networks))
    print("Saved Pandapower networks.")

    # Save node and edge features as a Pickle file
    with open(f"{save_location}/graph_features.pkl", "wb") as f:
        pkl.dump(features, f)
    print("Saved node and edge features.")

    print("Saving process completed successfully.")
