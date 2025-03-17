import os
import json
import pickle as pkl
from enum import Enum
import torch
import torch_geometric
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import pandapower as pp
from preprocess_data import * 

class DataloaderType(Enum):
    DEFAULT = "default"
    GRAPHYR = "graphyr"
    PINN = "pinn"


class DNRDataset(Data):
    """
    Custom PyG dataset for DNR data that handles advanced batching
    by custom increment and cat functions
    """ 
    def __init__(self, **kwargs):
        super(DNRDataset, self).__init__(**kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        """Custom incremental counting for correct batching."""
        if key == 'edge_index':
            return self.num_nodes
        elif key in ["conductance_matrix_index", "adjacency_matrix_index", 
                    "switch_matrix_index", "laplacian_matrix_index", 
                    "admittance_matrix_index"]:
            return self.num_nodes
        else:
            return 0
        
    def __cat_dim__(self, key, value, *args, **kwargs):
        """Return the dimension for which `value` will get concatenated for proper batching """
        # Edge indices are concatenated along the last dimension  
        if key in ["edge_index", "conductance_matrix_index", "adjacency_matrix_index", 
                  "switch_matrix_index", "laplacian_matrix_index", "admittance_matrix_index"]:
            return 1
        
        # Edge and Node attributes are concatenated along the first dimension
        elif key in [
            'x', 'edge_attr', 'conductance_matrix_values', 'adjacency_matrix_values',
            'switch_matrix_values', 'laplacian_matrix_values', 'admittance_matrix_values'
        ]:
            return 0
        
        # Scalar or graph-level features are concatenated along the batch dimension
        elif key in ['SBase', 'VBase', 'ZBase', 'YBase', 'IBase', 'vLow', 'vUpp']:
            return None
        else:
            return 0



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


def merge_features_into_nx(nx_graph, node_feats, edge_feats):
    if node_feats is None or edge_feats is None:
        raise ValueError("Missing node or edge features")
    
    # Add node features
    for node, feats in node_feats.items():
        if node not in nx_graph.nodes():
            continue
        for feat_key, feat_value in feats.items():
            nx_graph.nodes[node][feat_key] = feat_value
    
    # Add edge features
    for (u, v), e_feats in edge_feats.items():
        if not nx_graph.has_edge(u, v):
            continue
        for feat_key, feat_value in e_feats.items():
            nx_graph[u][v][feat_key] = feat_value

def check_missing_edge_features(nx_graph, edge_feats, print_results=True, default_values=None):
    """
    Check for missing edge features in a NetworkX graph and provide detailed statistics.
    
    Parameters:
    -----------
    nx_graph : NetworkX graph
        The graph to check for missing edge features
    edge_feats : dict
        Dictionary of edge features with keys as edge tuples (u, v) and values as dictionaries of features
    print_results : bool, optional
        Whether to print the results (default: True)
    default_values : dict, optional
        Dictionary of default values to use for missing features (default: None)
        
    Returns:
    --------
    dict
        Dictionary with detailed statistics about missing edge features
    """
    # Get all unique attribute names from the edge features
    all_edge_attrs = set()
    for feats in edge_feats.values():
        all_edge_attrs.update(feats.keys())
    
    if not all_edge_attrs:
        if print_results:
            print("No edge attributes found in the feature dictionary.")
        return {"error": "No edge attributes found"}
    
    # Set up default values if not provided
    if default_values is None:
        default_values = {
            "R": 0.01,
            "X": 0.01,
            "switch_state": 0.0
        }
    
    # Initialize counters
    total_edges = nx_graph.number_of_edges()
    if total_edges == 0:
        if print_results:
            print("Graph has no edges.")
        return {"error": "Graph has no edges"}
    
    # For each attribute, track:
    # - How many edges are missing this attribute
    # - Which specific edges are missing this attribute
    stats = {
        attr: {
            "missing_count": 0,
            "missing_edges": [],
            "default_value": default_values.get(attr, 0.0)
        } for attr in all_edge_attrs
    }
    
    # Check each edge
    is_directed = nx_graph.is_directed()
    for u, v in nx_graph.edges():
        edge = (u, v)
        
        # For undirected graphs, check both edge directions in the feature dictionary
        if not is_directed:
            reverse_edge = (v, u)
            feats = edge_feats.get(edge, edge_feats.get(reverse_edge, {}))
        else:
            feats = edge_feats.get(edge, {})
        
        # Check for missing attributes
        for attr in all_edge_attrs:
            if attr not in feats:
                stats[attr]["missing_count"] += 1
                stats[attr]["missing_edges"].append(edge)
    
    # Calculate percentages and prepare results
    results = {}
    for attr in all_edge_attrs:
        missing_count = stats[attr]["missing_count"]
        missing_percentage = (missing_count / total_edges) * 100
        
        results[attr] = {
            "missing_count": missing_count,
            "total_edges": total_edges,
            "missing_percentage": missing_percentage,
            "default_value": stats[attr]["default_value"],
            # Limit the number of edges in the output to avoid overwhelming output
            "example_missing_edges": stats[attr]["missing_edges"][:5] if stats[attr]["missing_edges"] else []
        }
    
    # Print results if requested
    if print_results:
        graph_name = nx_graph.name if hasattr(nx_graph, "name") and nx_graph.name else "Unnamed"
        print(f"\nEdge Attribute Analysis for Graph: {graph_name}")
        print(f"Total Edges: {total_edges}")
        print(f"Graph Type: {'Directed' if is_directed else 'Undirected'}")
        print("\nMissing Edge Attributes:")
        
        # Sort attributes by missing percentage (highest first)
        sorted_attrs = sorted(all_edge_attrs, key=lambda attr: results[attr]["missing_percentage"], reverse=True)
        
        for attr in sorted_attrs:
            res = results[attr]
            print(f"- {attr}:")
            print(f"  * Missing: {res['missing_count']}/{res['total_edges']} edges ({res['missing_percentage']:.2f}%)")
            print(f"  * Default value used: {res['default_value']}")
            
            if res["example_missing_edges"]:
                print(f"  * Example missing edges: {res['example_missing_edges'][:3]}...")
            print()
    
    # Add overall summary
    results["summary"] = {
        "total_edges": total_edges,
        "is_directed": is_directed,
        "attributes_analyzed": len(all_edge_attrs),
        "completely_missing_attributes": [
            attr for attr in all_edge_attrs 
            if results[attr]["missing_count"] == total_edges
        ]
    }
    
    return results
def check_and_fix_edge_attributes(nx_graph, print_details=True):
    """
    Checks for edge attribute inconsistencies and fixes them by ensuring all edges
    have exactly the same set of attributes with consistent types.
    
    Parameters:
    -----------
    nx_graph : NetworkX graph
        The graph to check and fix
    print_details : bool
        Whether to print detailed diagnostic information
        
    Returns:
    --------
    bool
        True if the graph was modified, False otherwise
    """
    if nx_graph.number_of_edges() == 0:
        if print_details:
            print("Graph has no edges.")
        return False
    
    # Collect all unique attributes from all edges
    all_attributes = set()
    attr_types = {}
    edge_attrs = {}
    
    # First pass: collect all attributes and their types
    for u, v, data in nx_graph.edges(data=True):
        edge = (u, v)
        edge_attrs[edge] = set(data.keys())
        all_attributes.update(data.keys())
        
        # Track attribute types
        for attr, value in data.items():
            if attr not in attr_types:
                attr_types[attr] = type(value)
            elif attr_types[attr] != type(value):
                attr_types[attr] = str  # Default to string for mixed types
    
    # Check if all edges have the same set of attributes
    is_consistent = True
    for edge, attrs in edge_attrs.items():
        if attrs != all_attributes:
            is_consistent = False
            break
    
    if print_details:
        print(f"\nEdge Attribute Consistency Check for Graph: {nx_graph.name if hasattr(nx_graph, 'name') else 'Unnamed'}")
        print(f"Total Edges: {nx_graph.number_of_edges()}")
        print(f"Total Unique Attributes: {len(all_attributes)}")
        print(f"All edges have identical attributes: {'Yes' if is_consistent else 'No'}")
        
        if not is_consistent:
            print("\nAttribute Distribution:")
            attr_counts = {attr: 0 for attr in all_attributes}
            for attrs in edge_attrs.values():
                for attr in attrs:
                    attr_counts[attr] += 1
            
            for attr, count in attr_counts.items():
                percentage = (count / nx_graph.number_of_edges()) * 100
                print(f"- {attr}: present in {count}/{nx_graph.number_of_edges()} edges ({percentage:.2f}%)")
            
            print("\nExample edges with different attribute sets:")
            attr_to_edge = {}
            for edge, attrs in edge_attrs.items():
                attr_key = frozenset(attrs)
                if attr_key not in attr_to_edge:
                    attr_to_edge[attr_key] = []
                if len(attr_to_edge[attr_key]) < 2:  # Keep just 2 examples
                    attr_to_edge[attr_key].append(edge)
            
            for i, (attr_set, edges) in enumerate(attr_to_edge.items(), 1):
                if i > 3:  # Limit to 3 different attribute sets
                    print("... more attribute sets exist ...")
                    break
                print(f"  Set {i}: {sorted(attr_set)}")
                for edge in edges:
                    print(f"    - Edge {edge}: {dict(nx_graph.edges[edge])}")
    
    # Fix the graph if inconsistent
    if not is_consistent:
        if print_details:
            print("\nFIXING: Adding missing attributes to ensure consistency...")
        
        modified = False
        # Define default values for common attributes
        default_values = {
            'R': 0.01,
            'X': 0.01,
            'switch_state': 0.0,
            'edge_idx': 0.0,
            'line_idx': 0.0
        }
        
        # Add missing attributes to all edges
        for u, v, data in nx_graph.edges(data=True):
            for attr in all_attributes:
                if attr not in data:
                    # Use default value if available, otherwise use a type-appropriate default
                    if attr in default_values:
                        data[attr] = default_values[attr]
                    else:
                        attr_type = attr_types.get(attr, float)
                        if attr_type == int:
                            data[attr] = 0
                        elif attr_type == float:
                            data[attr] = 0.0
                        elif attr_type == bool:
                            data[attr] = False
                        else:
                            data[attr] = ""
                    modified = True
        
        # Verify fix
        all_consistent = True
        for u, v, data in nx_graph.edges(data=True):
            if set(data.keys()) != all_attributes:
                all_consistent = False
                break
        
        if print_details:
            if all_consistent:
                print("FIX SUCCESSFUL: All edges now have the same attributes.")
            else:
                print("FIX FAILED: Edges still have inconsistent attributes.")
        
        return modified
    
    return False

def build_clean_graph(nx_graph, node_feats, edge_feats, print_details=False):
    """
    Builds a new clean graph with consistent attributes by only including
    the essential attributes needed for PyG conversion.
    
    Parameters:
    -----------
    nx_graph : NetworkX graph
        The original graph structure
    node_feats : dict
        Dictionary of node features
    edge_feats : dict
        Dictionary of edge features
    print_details : bool
        Whether to print diagnostic information
        
    Returns:
    --------
    NetworkX graph
        A new graph with consistent attributes
    """
    import networkx as nx
    
    # Define the essential attributes we need
    essential_node_attrs = ["p", "q", "v", "theta"]
    essential_edge_attrs = ["R", "X", "switch_state"]
    
    # Create a new graph of the same type as the original
    if nx_graph.is_directed():
        clean_graph = nx.DiGraph()
    else:
        clean_graph = nx.Graph()
    
    # Copy graph name if it exists
    if hasattr(nx_graph, "name"):
        clean_graph.name = nx_graph.name
    
    # Create a mapping from original node IDs to consecutive integers starting from 0
    node_mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    
    # Add all nodes with only the essential attributes
    for node in nx_graph.nodes():
        # Map to new consecutive integer ID
        new_node_id = node_mapping[node]
        
        # Initialize with default values
        node_attrs = {
            "p": 0.0,
            "q": 0.0,
            "v": 1.0,
            "theta": 0.0
        }
        
        # Update from original graph attributes if available
        for attr in essential_node_attrs:
            if attr in nx_graph.nodes[node]:
                node_attrs[attr] = nx_graph.nodes[node][attr]
        
        # Update from node_feats if available
        if node_feats and node in node_feats:
            for attr in essential_node_attrs:
                if attr in node_feats[node]:
                    node_attrs[attr] = node_feats[node][attr]
        
        # Add node to clean graph with new ID
        clean_graph.add_node(new_node_id, **node_attrs)
    
    # Add all edges with only the essential attributes
    for u, v in nx_graph.edges():
        # Map to new consecutive integer IDs
        new_u = node_mapping[u]
        new_v = node_mapping[v]
        
        # Initialize with default values
        edge_attrs = {
            "R": 0.01,
            "X": 0.01,
            "switch_state": 0.0
        }
        
        # Update from original graph attributes if available
        for attr in essential_edge_attrs:
            if attr in nx_graph[u][v]:
                edge_attrs[attr] = nx_graph[u][v][attr]
        
        # Update from edge_feats if available
        if edge_feats:
            # Try both (u, v) and (v, u) for undirected graphs
            edge = (u, v)
            reverse_edge = (v, u)
            
            if edge in edge_feats:
                for attr in essential_edge_attrs:
                    if attr in edge_feats[edge]:
                        edge_attrs[attr] = edge_feats[edge][attr]
            elif reverse_edge in edge_feats:
                for attr in essential_edge_attrs:
                    if attr in edge_feats[reverse_edge]:
                        edge_attrs[attr] = edge_feats[reverse_edge][attr]
        
        # Add edge to clean graph with new IDs
        clean_graph.add_edge(new_u, new_v, **edge_attrs)
    
    if print_details:
        print(f"\nClean Graph Builder:")
        print(f"Original graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
        print(f"Clean graph: {clean_graph.number_of_nodes()} nodes, {clean_graph.number_of_edges()} edges")
        print(f"Node attributes: {essential_node_attrs}")
        print(f"Edge attributes: {essential_edge_attrs}")
        print(f"Node IDs remapped to consecutive integers: 0-{clean_graph.number_of_nodes()-1}")
        
        # Verify attribute consistency
        is_consistent = True
        for u, v, data in clean_graph.edges(data=True):
            if set(data.keys()) != set(essential_edge_attrs):
                is_consistent = False
                break
                
        print(f"Edge attribute consistency: {'Yes' if is_consistent else 'No'}")
        
        # Print the first few edges to verify the structure
        print("\nSample edges from clean graph:")
        for i, (u, v, data) in enumerate(clean_graph.edges(data=True)):
            if i >= 3:  # Print only first 3 edges
                break
            print(f"  Edge ({u}, {v}): {data}")
    
    return clean_graph


def create_pyg_data_from_nx(nx_graph, features, loader_type=DataloaderType.DEFAULT):
    """
    Create a PyTorch Geometric Data object from a NetworkX graph.
    
    Parameters:
    -----------
    nx_graph : NetworkX graph
    features : dict
        Dictionary of features for this graph
    loader_type : DataloaderType
        Type of dataloader to create
        
    Returns:
    --------
    torch_geometric.data.Data
        PyG Data object or None if conversion failed
    """
    import torch
    from torch_geometric.utils import from_networkx
    
    # Extract node and edge features
    node_feats = features.get("node_features", {})
    edge_feats = features.get("edge_features", {})
    
    # Build a clean graph with only essential attributes and consecutive integer node IDs
    clean_graph = build_clean_graph(nx_graph, node_feats, edge_feats)
    
    # DIAGNOSTIC: Check if the clean graph is properly structured
    print(f"Checking graph before conversion:")
    print(f"- Graph type: {type(clean_graph)}")
    print(f"- Number of nodes: {clean_graph.number_of_nodes()}")
    print(f"- Number of edges: {clean_graph.number_of_edges()}")
    
    # Create the edge_index tensor directly from the clean graph
    edges = list(clean_graph.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"- Manually created edge_index shape: {edge_index.shape}")
    
    # Get edge attributes for all edges in the same order
    edge_attrs = []
    for u, v in edges:
        edge_attrs.append([
            clean_graph[u][v].get("R", 0.01),
            clean_graph[u][v].get("X", 0.01),
            clean_graph[u][v].get("switch_state", 0.0)
        ])
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    print(f"- Manually created edge_attr shape: {edge_attr.shape}")
    
    # Gather node features
    num_nodes = clean_graph.number_of_nodes()
    node_features = []
    for node_idx in range(num_nodes):  # Iterate over nodes in order
        node_data = clean_graph.nodes[node_idx]
        node_features.append([
            node_data.get("p", 0.0),
            node_data.get("q", 0.0),
            node_data.get("v", 1.0),
            node_data.get("theta", 0.0)
        ])
    x = torch.tensor(node_features, dtype=torch.float)
    print(x)
    print(f"- Manually created node features shape: {x.shape}")
    
    from torch_geometric.data import Data
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )
    print(f"Created PyG Data object: {data}")
    # Create our custom DNRDataset with the manual data
    custom_data = DNRDataset(
        x=data.x, 
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        num_nodes=data.num_nodes
    )

    # Add switch_mask and edge_y
    edge_count = edge_index.size(1)
    switch_state_column = 2  
    if edge_attr.size(0) > 0:
        custom_data.edge_y = edge_attr[:, switch_state_column].float()
    else:
        custom_data.edge_y = torch.zeros(edge_count, dtype=torch.float)
    
    # Add matrices based on loader type
    if loader_type != DataloaderType.DEFAULT:
        conductance_matrix = calculate_conductance_matrix(clean_graph)
        custom_data.conductance_matrix_index = conductance_matrix.coalesce().indices()
        custom_data.conductance_matrix_values = conductance_matrix.coalesce().values()
        
        # Add adjacency matrix
        adjacency_matrix = calculate_adjacency_matrix(clean_graph)
        custom_data.adjacency_matrix_index = adjacency_matrix.coalesce().indices()
        custom_data.adjacency_matrix_values = adjacency_matrix.coalesce().values()
        
        # Add switch matrix
        switch_matrix = calculate_switch_matrix(clean_graph)
        custom_data.switch_matrix_index = switch_matrix.coalesce().indices()
        custom_data.switch_matrix_values = switch_matrix.coalesce().values()
    
    # Add PINN-specific matrices
    if loader_type == DataloaderType.PINN:
        # Add Laplacian matrix
        laplacian_matrix = calculate_laplacian_matrix(clean_graph)
        custom_data.laplacian_matrix_index = laplacian_matrix.coalesce().indices()
        custom_data.laplacian_matrix_values = laplacian_matrix.coalesce().values()
        
        # Add admittance matrix
        admittance_matrix = calculate_admittance_matrix(clean_graph)
        custom_data.admittance_matrix_index = admittance_matrix.coalesce().indices()
        custom_data.admittance_matrix_values = admittance_matrix.coalesce().values()
    
    return custom_data

    
def create_pyg_dataset(base_directory, loader_type=DataloaderType.DEFAULT):
    nx_graphs, pp_networks, features = load_graph_data(base_directory)
    
    data_list = []
    successful_conversions = 0
    failed_conversions = 0
    
    for graph_name in nx_graphs.keys():
        if graph_name not in features:
            print(f"Skipping {graph_name} - missing features")
            continue
        
        print(f"\n--- Processing graph: {graph_name} ---")
        nx_graph = nx_graphs[graph_name]
        feature_dict = features[graph_name]
        
        try:
            data = create_pyg_data_from_nx(nx_graph, feature_dict, loader_type)
            if data is not None:
                data_list.append(data)
                successful_conversions += 1
                print(f"Successfully converted graph: {graph_name}")
            else:
                failed_conversions += 1
                print(f"Failed to convert graph: {graph_name} - returned None")
        except Exception as e:
            failed_conversions += 1
            import traceback
            print(f"Error creating PyG data for {graph_name}: {e}")
            print(traceback.format_exc())
    
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
    parser.add_argument("--base_dir", type=str,default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_data_set", help="Base directory containing the train/validation folders")
    parser.add_argument("--secondary_dir", type=str,default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_data_set_test",help="Secondary directory containing the test/validation folders")
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
                
        print("test_batch:", batch_test) 
        print(f"Test Batch size: {len(batch_test)}")
        