import os
import json
import pickle as pkl
from enum import Enum
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx
import numpy as np
import pandapower as pp
from preprocess_data import *
from pandapower import from_json, from_json_dict
 

class DataloaderType(Enum):
    DEFAULT = "default"
    GRAPHYR = "graphyr"
    PINN = "pinn"

class DNRDataset(Data):
    def __init__(self, **kwargs):
        super(DNRDataset, self).__init__(**kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.num_nodes
        elif key in ["conductance_matrix_index", "adjacency_matrix_index", 
                     "switch_matrix_index", "laplacian_matrix_index", 
                     "admittance_matrix_index"]:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["edge_index", "conductance_matrix_index", "adjacency_matrix_index", 
                   "switch_matrix_index", "laplacian_matrix_index", "admittance_matrix_index"]:
            return 1
        elif key in [
            'x', 'edge_attr', 'conductance_matrix_values', 'adjacency_matrix_values',
            'switch_matrix_values', 'laplacian_matrix_values', 'admittance_matrix_values'
        ]:
            return 0
        elif key in ['SBase', 'VBase', 'ZBase', 'YBase', 'IBase', 'vLow', 'vUpp']:
            return None
        else:
            return 0

    @property
    def conductance_matrix(self):
        """
        Reconstructs and returns the full conductance matrix as a dense tensor.
        """
        if hasattr(self, "conductance_matrix_index") and hasattr(self, "conductance_matrix_values"):
            import torch
            sparse_G = torch.sparse_coo_tensor(
                self.conductance_matrix_index,
                self.conductance_matrix_values,
                (self.num_nodes, self.num_nodes)
            )
            return sparse_G.to_dense()
        else:
            raise AttributeError("Conductance matrix is not set for this data instance.")

def load_graph_data(base_directory):
    print("\nLoading stored data...")
    
    # Load features
    features = {}
    features_dir = os.path.join(base_directory, "graph_features")
    if os.path.exists(features_dir):
        for filename in os.listdir(features_dir):
            if filename.endswith(".pkl"):
                file_path = os.path.join(features_dir, filename)
                with open(file_path, "rb") as f:
                    key = os.path.splitext(filename)[0]  # Remove .pkl extension
                    print(f"Loading feature: {key}")
                    features[key] = pkl.load(f)
    else:
        print(f"Warning: No graph features found in: {features_dir}")
    print(f"Loaded {len(features)} feature sets.")
    
    # Load NetworkX graphs
    nx_graphs = {}
    nx_dir = os.path.join(base_directory, "networkx_graphs")
    if os.path.exists(nx_dir):
        for filename in os.listdir(nx_dir):
            if filename.endswith(".pkl"):
                file_path = os.path.join(nx_dir, filename)
                with open(file_path, "rb") as f:
                    key = os.path.splitext(filename)[0]  # Remove .pkl extension
                    nx_graphs[key] = pkl.load(f)
    else:
        print(f"Warning: No NetworkX graphs found in: {nx_dir}")
    print(f"Loaded {len(nx_graphs)} NetworkX graphs.")
    
    # Load Pandapower networks
    pp_networks = {}
    pp_dir = os.path.join(base_directory, "pandapower_networks")
    if os.path.exists(pp_dir):
        for filename in os.listdir(pp_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(pp_dir, filename)
                try:
                    with open(file_path, "r") as f:
                        key = os.path.splitext(filename)[0]  # Remove .json extension
                        json_str = f.read()
                        try:
                            pp_net = pp.from_json_string(json_str)
                            pp_networks[key] = pp_net
                        except Exception as e:
                            print(f"Warning: Could not load {key} as pandapower network: {e}")
                            pp_networks[key] = json.loads(json_str)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    else:
        print(f"Warning: No pandapower networks found in: {pp_dir}")
    print(f"Loaded {len(pp_networks)} Pandapower networks.")
    
    return nx_graphs, pp_networks, features

def create_pyg_data_from_nx(nx_graph, pp_network, loader_type=DataloaderType.DEFAULT,
                            use_fallback_features=False, fallback_features=None):
    # Validate that the graph has all required node attributes
    for node in nx_graph.nodes():
        for attr in ["p", "q", "v", "theta"]:
            if attr not in nx_graph.nodes[node]:
                if use_fallback_features and fallback_features and "node_features" in fallback_features:
                    node_feats = fallback_features["node_features"]
                    if node in node_feats and attr in node_feats[node]:
                        nx_graph.nodes[node][attr] = node_feats[node][attr]
                    else:
                        raise ValueError(f"Node {node} missing attribute '{attr}' and not found in fallback features")
                else:
                    raise ValueError(f"Node {node} missing required attribute '{attr}'")

    # Validate that the graph has all required edge attributes
    for u, v in nx_graph.edges():
        for attr in ["R", "X", "switch_state"]:
            if attr not in nx_graph[u][v]:
                if use_fallback_features and fallback_features and "edge_features" in fallback_features:
                    edge_feats = fallback_features["edge_features"]
                    if (u, v) in edge_feats and attr in edge_feats[(u, v)]:
                        nx_graph[u][v][attr] = edge_feats[(u, v)][attr]
                    elif (v, u) in edge_feats and attr in edge_feats[(v, u)]:
                        nx_graph[u][v][attr] = edge_feats[(v, u)][attr]
                    else:
                        raise ValueError(f"Edge {u}-{v} missing attribute '{attr}' and not found in fallback features")
                else:
                    raise ValueError(f"Edge {u}-{v} missing required attribute '{attr}'")

            if attr in ["R", "X"] and nx_graph[u][v][attr] == 0:
                raise ValueError(f"Edge {u}-{v} has zero {attr} value which will cause division by zero")

    # Create the edge_index tensor
    edges = list(nx_graph.edges())
    if not edges:
        raise ValueError("Graph has no edges")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Extract edge attributes
    edge_attrs = []
    for u, v in edges:
        edge_attrs.append([
            nx_graph[u][v]["R"],
            nx_graph[u][v]["X"],
            nx_graph[u][v]["switch_state"]
        ])
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # print("before loading:", pp_network[:100])
    # print(type(pp_network))
    # # Ensure pp_network is a dictionary before calling from_json_dict
    # if isinstance(pp_network, str):
    #     pp_network = json.loads(pp_network)
    # pp_network_loaded = from_json_dict(pp_network)
    # print("after loading:", str(pp_network_loaded)[:100])
    # print(pp_network_loaded.line.iloc[0])

    # line_currents = torch.tensor(pp_network_loaded.res_line.loading_percent.values, dtype=torch.float)
    # edge_attr = torch.cat([edge_attr, line_currents.unsqueeze(1)], dim=1)

    # Extract node features
    num_nodes = nx_graph.number_of_nodes()
    node_features = []
    nodes = list(nx_graph.nodes())
    if set(nodes) != set(range(len(nodes))):
        raise ValueError("Graph nodes must be consecutive integers starting from 0")
    for node_idx in range(num_nodes):
        node_data = nx_graph.nodes[node_idx]
        node_features.append([
            node_data["p"],
            node_data["q"],
            node_data["v"],
            node_data["theta"]
        ])
    x = torch.tensor(node_features, dtype=torch.float)

    # Create the PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )

    # Create our custom DNRDataset with the data
    custom_data = DNRDataset(
        x=data.x, 
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        num_nodes=data.num_nodes
    )

    # Add edge_y (switch state is the 3rd column - index 2)
    switch_state_column = 2
    custom_data.edge_y = edge_attr[:, switch_state_column].float()

    # Add matrices based on loader type
    if loader_type != DataloaderType.DEFAULT:
        conductance_matrix = calculate_conductance_matrix(nx_graph)
        coalesced = conductance_matrix.coalesce()
        custom_data.conductance_matrix_index = coalesced.indices()
        custom_data.conductance_matrix_values = coalesced.values()

        adjacency_matrix = calculate_adjacency_matrix(nx_graph)
        custom_data.adjacency_matrix_index = adjacency_matrix.coalesce().indices()
        custom_data.adjacency_matrix_values = adjacency_matrix.coalesce().values()

        switch_matrix = calculate_switch_matrix(nx_graph)
        custom_data.switch_matrix_index = switch_matrix.coalesce().indices()
        custom_data.switch_matrix_values = switch_matrix.coalesce().values()

    # Add PINN-specific matrices
    if loader_type == DataloaderType.PINN:
        laplacian_matrix = calculate_laplacian_matrix(nx_graph)
        custom_data.laplacian_matrix_index = laplacian_matrix.coalesce().indices()
        custom_data.laplacian_matrix_values = laplacian_matrix.coalesce().values()

        admittance_matrix = calculate_admittance_matrix(nx_graph)
        custom_data.admittance_matrix_index = admittance_matrix.coalesce().indices()
        custom_data.admittance_matrix_values = admittance_matrix.coalesce().values()

    return custom_data
    
def create_pyg_dataset(base_directory, loader_type=DataloaderType.DEFAULT, use_fallback_features=False):
    nx_graphs, pp_networks, features = load_graph_data(base_directory)
    
    data_list = []
    successful_conversions = 0
    failed_conversions = 0
    
    for graph_name in nx_graphs.keys():
        print(f"\n--- Processing graph: {graph_name} ---")
        nx_graph = nx_graphs[graph_name]
        
        # Get features as fallback only if requested
        fallback_features = features.get(graph_name, None) if use_fallback_features else None
        
        try:
            # Create PyG data directly from the NetworkX graph, let errors propagate
            data = create_pyg_data_from_nx(
                nx_graph, 
                pp_networks[graph_name],
                loader_type, 
                use_fallback_features=use_fallback_features,
                fallback_features=fallback_features
            )
            
            data_list.append(data)
            successful_conversions += 1
            print(f"Successfully converted graph: {graph_name}")
                
        except Exception as e:
            failed_conversions += 1
            print(f"Error creating PyG data for {graph_name}: {e}")
            # Let the exception propagate if this is a critical error
            if "missing required attribute" in str(e) or "zero" in str(e):
                raise  # Re-raise important errors
    
    print(f"\nCreated {len(data_list)} PyG data objects")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    
    return data_list


def create_dynamic_loader(dataset, max_nodes=1000, max_edges=5000, shuffle=True, **kwargs):
    class DynamicBatchSampler(torch.utils.data.Sampler):
        def __init__(self, dataset, max_nodes, max_edges, shuffle):
            self.dataset = dataset
            self.max_nodes = max_nodes
            self.max_edges = max_edges
            self.shuffle = shuffle
            self.indices = list(range(len(dataset)))
            
            # Pre-compute the number of nodes and edges for each graph
            self.graph_sizes = []
            for data in dataset:
                self.graph_sizes.append((data.num_nodes, data.num_edges))
            
        def __iter__(self):
            # Get indices of all graphs
            indices = self.indices.copy()
            
            # Shuffle if required
            if self.shuffle:
                torch.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
                torch.randperm(len(indices), out=torch.tensor(indices))
            
            # Create batches
            current_batch = []
            current_nodes = 0
            current_edges = 0
            
            for idx in indices:
                nodes, edges = self.graph_sizes[idx]
                
                if ((current_nodes + nodes > self.max_nodes or 
                     current_edges + edges > self.max_edges) and 
                    len(current_batch) > 0):
                    # Yield the current batch
                    yield current_batch
                    # Start a new batch with the current graph
                    current_batch = [idx]
                    current_nodes = nodes
                    current_edges = edges
                else:
                    # Add this graph to the current batch
                    current_batch.append(idx)
                    current_nodes += nodes
                    current_edges += edges
            
            # Yield the last batch if it's not empty
            if current_batch:
                yield current_batch
                
        def __len__(self):
            # This is an estimate since actual number of batches depends on graph sizes
            if not self.graph_sizes:
                return 0
            total_nodes = sum(nodes for nodes, _ in self.graph_sizes)
            total_edges = sum(edges for _, edges in self.graph_sizes)
            return max(1, min(
                int(total_nodes / self.max_nodes) + 1,
                int(total_edges / self.max_edges) + 1
            ))
    
    # Create the batch sampler
    batch_sampler = DynamicBatchSampler(dataset, max_nodes, max_edges, shuffle)
    
    # Create DataLoader with the custom batch sampler
    return DataLoader(
        dataset, 
        batch_sampler=batch_sampler, 
        **kwargs
    )


def create_data_loaders(base_directory,secondary_directory=None, loader_type=DataloaderType.DEFAULT, 
                       batch_size=32, max_nodes=1000, max_edges=5000,
                        transform=None, train_ratio=0.8, seed=0,batching_type="standard", num_workers=1,):
    dataset = create_pyg_dataset(base_directory, loader_type)
    if secondary_directory:
        print("==================================================","\n start loading secondary data")
        val_real_set = create_pyg_dataset(os.path.join(secondary_directory, "validation"), loader_type)
        test_set = create_pyg_dataset(os.path.join(secondary_directory, "test"), loader_type)

    
    if transform:
        dataset = [transform(data) for data in dataset]
        if secondary_directory: 
            val_real_set = [transform(data) for data in val_real_set]
            test_set = [transform(data) for data in test_set]
    
    torch.manual_seed(seed)
    train_size = int(train_ratio * len(dataset))
    train_set, val_synthetic_set= torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
     
    print("==================================================","\n start creating loaders")
    val_real_loader, test_loader= [None], [None]

    # Create loaders based on the loader type
    if batching_type == "dynamic":
        train_loader = create_dynamic_loader(train_set, max_nodes=max_nodes, max_edges=max_edges, shuffle=True, num_workers=num_workers)
        val_synthetic_loader = create_dynamic_loader(val_synthetic_set, max_nodes=max_nodes, max_edges=max_edges, shuffle=False, num_workers=num_workers)
        if secondary_directory:
            val_real_loader = create_dynamic_loader(val_real_set, max_nodes=max_nodes, max_edges=max_edges, shuffle=False, num_workers=num_workers)
            test_loader = create_dynamic_loader(test_set, max_nodes=max_nodes, max_edges=max_edges, shuffle=False, num_workers=num_workers)
    else:
        # Use standard DataLoader
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=num_workers)
        val_synthetic_loader = DataLoader(val_synthetic_set, batch_size=batch_size, shuffle=False,num_workers=num_workers)
        if secondary_directory:
            val_real_loader = DataLoader(val_real_set, batch_size=batch_size, shuffle=False,num_workers=num_workers)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    

    print(f"Created data loaders with: \n       training: {len(train_set)}\n        synthetic validation:{len(val_synthetic_loader)}\n      real validation:{len(val_real_loader)}\n         test samples:{len(test_loader)}")
    
    return train_loader, val_synthetic_loader, val_real_loader, test_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create data loaders for power network data")
    parser.add_argument("--base_dir", type=str,default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\transformed_subgraphs_24032025", help="Base directory containing the train/validation folders")
    parser.add_argument("--secondary_dir", type=str, #default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_data_set_test",
                        help="Secondary directory containing the test/validation folders")
    parser.add_argument("--loader_type", type=str, default="pinn", 
                        choices=["default", "graphyr", "pinn",],
                        help="Type of dataloader to create")
    parser.add_argument("--batching_type", type=str, default="dynamic",
                        choices =["standard", "dynamic"])
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_nodes", type=int, default=1000, help="Maximum number of nodes in a batch (for dynamic batching)")
    parser.add_argument("--max_edges", type=int, default=5000, help="Maximum number of edges in a batch (for dynamic batching)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of training set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    # Map string argument to enum
    loader_type_map = {
        "default": DataloaderType.DEFAULT,
        "graphyr": DataloaderType.GRAPHYR,
        "pinn": DataloaderType.PINN,
    }
    loader_type = loader_type_map[args.loader_type]
    
    # Create data loaders
    train_loader, val_synthetic_loader,val_real_loader, test_loader = create_data_loaders(
        base_directory=args.base_dir,
        secondary_directory=args.secondary_dir,
        loader_type=loader_type,
        batch_size=args.batch_size,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        batching_type = args.batching_type
    )
    
    print("\nData loaders created successfully.")

    # Print sample batch information
    if train_loader:
        print("\nSample batch information:")
        if isinstance(train_loader, list):
            # For NeighborLoader
            batch = next(iter(train_loader[0]))
            batch_test = next(iter(test_loader[0]))
        else:
            # For regular DataLoader
            batch = next(iter(train_loader))
            batch_test = next(iter(test_loader))
        print(f"Batch type: {type(batch)}")
        print("batch:", batch)
        print(f"Batch size: {len(batch)}")
        # Print loader-specific features
        if loader_type in [DataloaderType.GRAPHYR, DataloaderType.PINN]:
            if hasattr(batch, 'conductance_matrix_index'):
                print(f"Conductance matrix indices shape: {batch.conductance_matrix_index.shape}")
            if hasattr(batch, 'adjacency_matrix_index'):
                print(f"Adjacency matrix indices shape: {batch.adjacency_matrix_index.shape}")
            if hasattr(batch, 'switch_matrix_index'):
                print(f"Switch matrix indices shape: {batch.switch_matrix_index.shape}")
        
        if loader_type == DataloaderType.PINN:
            if hasattr(batch, 'laplacian_matrix_index'):
                print(f"Laplacian matrix indices shape: {batch.laplacian_matrix_index.shape}")
            if hasattr(batch, 'admittance_matrix_index'):
                print(f"Admittance matrix indices shape: {batch.admittance_matrix_index.shape}")
        if args.secondary_dir:         
            print("test_batch:", batch_test) 
            print(f"Test Batch size: {len(batch_test)}")
        