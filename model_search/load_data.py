import os
import json
import pickle as pkl
from pathlib import Path
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
from pandapower.pypower.makeYbus import makeYbus
import numpy as np 

def load_graph_data(folder_path):
    folder_path = Path(folder_path)
    graph_features_path = folder_path / "graph_features.pkl"
    nx_graphs_path = folder_path / "networkx_graphs.pkl"
    pp_networks_path = folder_path / "pandapower_networks.json"
    
    with open(graph_features_path, "rb") as f:
        graph_features = pkl.load(f)
    with open(nx_graphs_path, "rb") as f:
        nx_graphs = pkl.load(f)
    with open(pp_networks_path, "r") as f:
        pp_networks = json.load(f)
    return nx_graphs, graph_features, pp_networks

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=1):
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

def unify_node_attributes(nx_graph, required_node_attrs):
    for node in nx_graph.nodes():
        for attr in list(nx_graph.nodes[node].keys()):
            if attr not in required_node_attrs:
                del nx_graph.nodes[node][attr]
        for attr in required_node_attrs:
            if attr not in nx_graph.nodes[node]:
                nx_graph.nodes[node][attr] = 0.0

def unify_edge_attributes(nx_graph, required_edge_attrs):
    for u, v in nx_graph.edges():
        for attr in list(nx_graph[u][v].keys()):
            if attr not in required_edge_attrs:
                del nx_graph[u][v][attr]
        for attr in required_edge_attrs:
            if attr not in nx_graph[u][v]:
                nx_graph[u][v][attr] = 0.0

def merge_features_into_nx(nx_graph, node_feats, edge_feats):
    if node_feats is None or edge_feats is None:
        raise ValueError("Missing node or edge features")
    for node, feats in node_feats.items():
        if node not in nx_graph.nodes():
            continue
        for feat_key, feat_value in feats.items():
            nx_graph.nodes[node][feat_key] = feat_value
    for (u, v), e_feats in edge_feats.items():
        if not nx_graph.has_edge(u, v):
            continue
        for feat_key, feat_value in e_feats.items():
            nx_graph[u][v][feat_key] = feat_value

def create_pyg_data(nx_graph, pp_net):
    # Make sure required attributes exist in the graph
    unify_node_attributes(nx_graph, ["p", "q", "v", "theta"])
    unify_edge_attributes(nx_graph, ["R", "X", "switch_state"])

    # Print node and edge count before conversion
    print(f"NetworkX graph has {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")
    
    # Convert networkx to PyG
    data = from_networkx(
        nx_graph,
        group_node_attrs=["p", "q", "v", "theta"],
        group_edge_attrs=["R", "X", "switch_state"]
    )

    # Debug the data structure after conversion
    print(f"PyG data has {data.num_nodes} nodes and {data.num_edges} edges")
    print("Available node attributes:", [key for key in data.keys() if isinstance(data[key], torch.Tensor) and data[key].size(0) == data.num_nodes])
    print("Available edge attributes:", [key for key in data.keys() if isinstance(data[key], torch.Tensor) and data[key].size(0) == data.num_edges])
    
    # Process node features
    if hasattr(data, 'p') and hasattr(data, 'q') and hasattr(data, 'v') and hasattr(data, 'theta'):
        # If they exist as separate attributes
        data.x = torch.stack([data.p, data.q, data.v, data.theta], dim=-1).float()
        del data.p, data.q, data.v, data.theta
    elif hasattr(data, 'x') and data.x.size(-1) == 4:
        # If they've been automatically grouped into x
        pass
    else:
        # If the structure is different than expected, create default features
        print(f"Warning: Node features not found in expected format, creating zeros")
        data.x = torch.zeros((data.num_nodes, 4), dtype=torch.float32)
    
    # Process edge features
    if hasattr(data, 'R') and hasattr(data, 'X'):
        data.edge_attr = torch.stack([data.R, data.X], dim=-1).float()
        del data.R, data.X
    elif hasattr(data, 'edge_attr') and data.edge_attr.size(-1) >= 2:
        # If they've been automatically grouped
        if data.edge_attr.size(-1) > 2:
            # Only keep the first two features (R and X)
            data.edge_attr = data.edge_attr[:, :2].float()
    else:
        print(f"Warning: Edge features not found in expected format, creating zeros")
        data.edge_attr = torch.zeros((data.num_edges, 2), dtype=torch.float32)
    
    # Clean up switch_state if it exists
    if hasattr(data, 'switch_state'):
        del data.switch_state

    conductance_matrix = compute_conductance_matrix(nx_graph, data.num_nodes)
    data.conductance_matrix = conductance_matrix

    data._store_attr = ["conductance_matrix"]
    
    # Add line currents and net injection placeholders for physics calculations
    data.line_currents = torch.zeros(data.num_edges, dtype=torch.float32)
    data.net_injection = torch.zeros(data.num_nodes, dtype=torch.float32)
    
    # Create safe version of compute_switch_matrix 
    #data.switch_matrix = safe_compute_switch_matrix(nx_graph, data.num_nodes)
    
    # Create safe adjacency matrix
    # try:
    #     # Get node mapping from networkx to PyG (important!)
    #     if hasattr(data, '_mapping'):
    #         node_mapping = data._mapping
    #         # Create adjacency matrix with the correct indices
    #         adj_matrix = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.float32)
            
    #         for u, v in nx_graph.edges():
    #             # Map original indices to PyG indices
    #             if u in node_mapping and v in node_mapping:
    #                 u_idx, v_idx = node_mapping[u], node_mapping[v]
    #                 if u_idx < data.num_nodes and v_idx < data.num_nodes:
    #                     adj_matrix[u_idx, v_idx] = 1.0
    #                     adj_matrix[v_idx, u_idx] = 1.0
            
    #         data.adjacency_matrix = adj_matrix
    #     else:
    #         # Fallback to using the numpy array method, but be careful with indices
    #         adj_np = nx.to_numpy_array(nx_graph)
    #         # Ensure the matrix size matches the number of nodes in PyG data
    #         if adj_np.shape[0] > data.num_nodes:
    #             adj_np = adj_np[:data.num_nodes, :data.num_nodes]
    #         elif adj_np.shape[0] < data.num_nodes:
    #             # Pad with zeros if needed
    #             new_adj = np.zeros((data.num_nodes, data.num_nodes))
    #             new_adj[:adj_np.shape[0], :adj_np.shape[1]] = adj_np
    #             adj_np = new_adj
                
    #         data.adjacency_matrix = torch.tensor(adj_np, dtype=torch.float32)
    # except Exception as e:
    #     print(f"Error creating adjacency matrix: {e}")
    #     data.adjacency_matrix = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.float32)
    print(data)
    return data

def safe_compute_switch_matrix(nx_graph, num_nodes):
    """Create a switch matrix that handles index mapping safely"""
    switch_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    
    # Get a mapping of networkx node IDs to consecutive indices
    # This is needed because networkx can have non-consecutive node IDs
    node_list = list(nx_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    try:
        for u, v, data in nx_graph.edges(data=True):
            if "switch_state" in data:
                # Check if indices are within bounds
                u_idx, v_idx = node_to_idx.get(u), node_to_idx.get(v)
                if u_idx is not None and v_idx is not None:
                    if u_idx < num_nodes and v_idx < num_nodes:
                        switch_matrix[u_idx, v_idx] = data["switch_state"]
                        switch_matrix[v_idx, u_idx] = data["switch_state"]
    except Exception as e:
        print(f"Error in computing switch matrix: {e}")
    
    return switch_matrix

def create_pyg_dataset(folder_path):
    nx_graphs, graph_features, pp_nets = load_graph_data(folder_path)
    data_list = []
    skipped = 0
    total = len(nx_graphs)

    for graph_name, nx_graph in nx_graphs.items():
        print("graph_name", graph_name) 
        if nx_graph is None:
            print(f"Skipping {graph_name}: Graph is None.")
            skipped += 1
            continue
        if graph_name not in graph_features:
            print(f"Skipping {graph_name}: Missing graph features.")
            skipped += 1
            continue

        features_dict = graph_features[graph_name]
        node_feats = features_dict.get("node_features")
        edge_feats = features_dict.get("edge_features")

        if node_feats is None or edge_feats is None:
            print(f"Skipping {graph_name}: Missing node or edge features.")
            skipped += 1
            continue

        try:
            merge_features_into_nx(nx_graph, node_feats, edge_feats)
        except Exception as e:
            print(f"Skipping {graph_name}: Error merging features: {e}")
            skipped += 1
            continue

        if graph_name not in pp_nets:
            print(f"Skipping {graph_name}: No corresponding pandapower network.")
            skipped += 1
            continue

        try:
            pp_net = pp_nets[graph_name]
            #print(pp_net)
            data = create_pyg_data(nx_graph, pp_net)
        except Exception as e:
            print(f"Skipping {graph_name}: Error creating PyG data: {e}")
            skipped += 1
            continue

        data_list.append(data)

    print(f"Total graphs: {total}; Skipped: {skipped}; Processed: {len(data_list)}")
    return data_list


def get_pyg_loader(folder_path, batch_size=4, shuffle=True, transform=None):
    data_list = create_pyg_dataset(folder_path)
    
    if transform:
        data_list = [transform(data) for data in data_list]

    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

def combined_augmentations(data):
    data = normalize_features(data)
    data = gaussian_noise_injection(data, noise_std=0.01)
    data = switch_state_augmentation(data, flip_prob=0.1)
    data = domain_specific_feature_augmentation(data)
    data = feature_dropout(data, dropout_prob=0.2)
    data = global_node_addition(data)
    
    return data
def compute_conductance_matrix(nx_graph, num_nodes):
    """Compute the conductance matrix from impedance values in the graph"""
    conductance_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    
    # Get a mapping of networkx node IDs to consecutive indices
    node_list = list(nx_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    try:
        for u, v, data in nx_graph.edges(data=True):
            if "R" in data and "X" in data:
                # Get the indices
                u_idx, v_idx = node_to_idx.get(u), node_to_idx.get(v)
                if u_idx is not None and v_idx is not None:
                    if u_idx < num_nodes and v_idx < num_nodes:
                        # Calculate conductance from resistance (ignore reactance for simplicity)
                        # In a more complete implementation, you'd use complex admittance
                        r_value = data["R"]
                        x_value = data["X"]
                        
                        # Avoid division by zero
                        if r_value > 0:
                            g_value = 1.0 / r_value
                            # Fill the conductance matrix (symmetric)
                            conductance_matrix[u_idx, v_idx] = g_value
                            conductance_matrix[v_idx, u_idx] = g_value
                            # Diagonal elements (negative sum of row)
                            conductance_matrix[u_idx, u_idx] -= g_value
                            conductance_matrix[v_idx, v_idx] -= g_value
    except Exception as e:
        print(f"Error in computing conductance matrix: {e}")
    
    return conductance_matrix


if __name__ == "__main__":
    folder_path = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\1741471046.04005"
    loader = get_pyg_loader(folder_path)
    for data in loader:
        print(data)