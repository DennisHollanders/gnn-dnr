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
    unify_node_attributes(nx_graph, ["p", "q", "v", "theta"])
    unify_edge_attributes(nx_graph, ["R", "X", "switch_state"])

    data = from_networkx(
        nx_graph,
        group_node_attrs=["p", "q", "v", "theta"],
        group_edge_attrs=["R", "X", "switch_state"]
    )

    # Convert features to tensors
    data.x = torch.stack([data.p, data.q, data.v, data.theta], dim=-1).float()
    data.edge_attr = torch.stack([data.R, data.X], dim=-1).float()
    
    del data.p, data.q, data.v, data.theta
    del data.R, data.X

    print("ppnet \n ", pp_net, "\n \n ")

    # Ensure valid pp_net before computing Ybus
    #if pp_net is None:
    #    raise ValueError("pp_net is None, cannot compute Ybus!")

    #try:
    _, Ymatrix, _ = makeYbus(pp_net)
    data.conductance_matrix = torch.tensor(np.real(Ymatrix.toarray()), dtype=torch.float32)
    #except Exception as e:
    #    print(f"Error computing conductance matrix for graph: {e}")
    #    data.conductance_matrix = torch.zeros((len(nx_graph.nodes), len(nx_graph.nodes)), dtype=torch.float32)

    data.adjacency_matrix = torch.tensor(nx.to_numpy_array(nx_graph), dtype=torch.float32)
    data.switch_matrix = compute_switch_matrix(nx_graph)

    return data

def create_pyg_dataset(folder_path):
    nx_graphs, graph_features, pp_nets = load_graph_data(folder_path)
    data_list = []
    skipped = 0
    total = len(nx_graphs)

    for graph_name, nx_graph in nx_graphs.items():
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
            data = create_pyg_data(nx_graph, pp_nets[graph_name])
        except Exception as e:
            print(f"Skipping {graph_name}: Error creating PyG data: {e}")
            skipped += 1
            continue

        data_list.append(data)

    print(f"Total graphs: {total}; Skipped: {skipped}; Processed: {len(data_list)}")
    return data_list


def get_pyg_loader(folder_path, batch_size=4, shuffle=True,transform=None):
    data_list = create_pyg_dataset(folder_path)
    if transform:
        dataset = [transform(data) for data in dataset]
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

def compute_switch_matrix(nx_graph):
    num_nodes = len(nx_graph.nodes)
    switch_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for u, v, data in nx_graph.edges(data=True):
        if "switch_state" in data:
            switch_matrix[u, v] = data["switch_state"]
            switch_matrix[v, u] = data["switch_state"]

    return switch_matrix

def combined_augmentations(data):
    data = normalize_features(data)
    data = gaussian_noise_injection(data, noise_std=0.01)
    data = switch_state_augmentation(data, flip_prob=0.1)
    data = domain_specific_feature_augmentation(data)
    data = feature_dropout(data, dropout_prob=0.2)
    data = global_node_addition(data)
    
    return data