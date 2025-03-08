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

def create_pyg_data(nx_graph):
    unify_node_attributes(nx_graph, ["p", "q", "v", "theta"])
    unify_edge_attributes(nx_graph, ["R", "X", "switch_state"])
    data = from_networkx(
        nx_graph,
        group_node_attrs=["p", "q", "v", "theta"],
        group_edge_attrs=["R", "X", "switch_state"]
    )
    if hasattr(data, "p"):
        data.x = torch.stack([data.p, data.q, data.v, data.theta], dim=-1).float()
        del data.p, data.q, data.v, data.theta
    if hasattr(data, "R"):
        data.edge_attr = torch.stack([data.R, data.X], dim=-1).float()
        del data.R, data.X
    if hasattr(data, "switch_state"):
        data.edge_y = data.switch_state.view(-1).float()
        del data.switch_state
    return data

def create_pyg_dataset(folder_path):
    nx_graphs, graph_features, _ = load_graph_data(folder_path)
    data_list = []
    skipped = 0
    total = len(nx_graphs)
    for graph_name, nx_graph in nx_graphs.items():
        if nx_graph is None:
            skipped += 1
            continue
        if graph_name not in graph_features:
            skipped += 1
            continue
        features_dict = graph_features[graph_name]
        node_feats = features_dict.get("node_features")
        edge_feats = features_dict.get("edge_features")
        if node_feats is None or edge_feats is None:
            skipped += 1
            continue
        try:
            merge_features_into_nx(nx_graph, node_feats, edge_feats)
        except Exception:
            skipped += 1
            continue
        try:
            data = create_pyg_data(nx_graph)
        except Exception:
            skipped += 1
            continue
        data_list.append(data)
    print(f"Total graphs: {total}; Skipped: {skipped}; Processed: {len(data_list)}")
    return data_list

def get_pyg_loader(folder_path, batch_size=4, shuffle=True):
    data_list = create_pyg_dataset(folder_path)
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)