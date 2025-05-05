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
import tqdm
import logging 
import sys
import random

# Add necessary source paths
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from electrify_subgraph import extract_node_features, extract_edge_features


# ─── configure root logger ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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

def load_graph_data_old(base_directory):
    logger.info("Loading stored data from %s", base_directory)

    # Load features
    features = {}
    features_dir = os.path.join(base_directory, "graph_features")
    if os.path.isdir(features_dir):
        for fn in os.listdir(features_dir):
            if not fn.endswith(".pkl"): 
                continue
            key = fn[:-4]
            path = os.path.join(features_dir, fn)
            logger.debug("  → loading feature %s", key)
            with open(path, "rb") as f:
                features[key] = pkl.load(f)
    else:
        logger.warning("No graph_features folder at %s", features_dir)
    logger.info("Loaded %d feature sets", len(features))

    # Load NetworkX
    nx_graphs = {}
    nx_dir = os.path.join(base_directory, "networkx_graphs")
    if os.path.isdir(nx_dir):
        for fn in os.listdir(nx_dir):
            if not fn.endswith(".pkl"): 
                continue
            key = fn[:-4]
            path = os.path.join(nx_dir, fn)
            try:
                with open(path, "rb") as f:
                    nx_graphs[key] = pkl.load(f)
            except Exception as e:
                logger.error("Failed loading NX graph %s: %s", key, e)
    else:
        logger.warning("No networkx_graphs folder at %s", nx_dir)
    logger.info("Loaded %d NetworkX graphs", len(nx_graphs))

    # Load pandapower
    pp_networks = {}
    pp_dir = os.path.join(base_directory, "pandapower_networks")
    if os.path.isdir(pp_dir):
        for fn in os.listdir(pp_dir):
            if not fn.endswith(".json"): 
                continue
            key = fn[:-5]
            path = os.path.join(pp_dir, fn)
            try:
                with open(path) as f:
                    raw = f.read()
                try:
                    pp_networks[key] = pp.from_json_string(raw)
                except Exception:
                    pp_networks[key] = json.loads(raw)
                logger.debug("  → loaded pandapower network %s", key)
            except Exception as e:
                logger.error("Failed loading pandapower %s: %s", key, e)
    else:
        logger.warning("No pandapower_networks folder at %s", pp_dir)
    logger.info("Loaded %d Pandapower networks", len(pp_networks))

    return nx_graphs, pp_networks, features

# def load_graph_data(base_directory):
#     """
#     Load networkx graphs, pandapower networks and feature‐pickles
#     from three subfolders: original, post_MST, optimization.
#     Returns three dicts: nx_graphs, pp_networks, features each keyed by
#     phase ∈ {"original","post_mst","optimization"} → {graph_id → obj}
#     """
#     phases = {
#     "original":    ("original",     "_original"),
#     "post_mst":    ("post_MST",     "_post_mst"),
#     "optimization":("optimized",    "_optimized"), 
#     }

#     #nx_graphs   = {p: {} for p in phases}
#     pp_networks = {p: {} for p in phases}
#     #features    = {p: {} for p in phases}

#     for phase, (subdir, suffix) in phases.items():
#         # feat_dir = os.path.join(base_directory, subdir, "graph_features")
#         # for fn in os.listdir(feat_dir) if os.path.isdir(feat_dir) else []:
#         #     if fn.endswith(".pkl"):
#         #         base, _ = os.path.splitext(fn)
#         #         gid = base[:-len(suffix)] if base.endswith(suffix) else base
#         #         with open(os.path.join(feat_dir, fn), "rb") as f:
#         #             features[phase][gid] = pkl.load(f)
#         # print(f"Loaded {len(features[phase])} {phase} features")

#         # nx_dir = os.path.join(base_directory, subdir, "networkx_graphs")
#         # for fn in os.listdir(nx_dir) if os.path.isdir(nx_dir) else []:
#         #     base, ext = os.path.splitext(fn)
#         #     gid = base[:-len(suffix)] if base.endswith(suffix) else base
#         #     path = os.path.join(nx_dir, fn)
#         #     if ext == ".pkl":
#         #         with open(path, "rb") as f:
#         #             nx_graphs[phase][gid] = pkl.load(f)
#         #     elif ext == ".graphml":
#         #         nx_graphs[phase][gid] = nx.read_graphml(path)
#         # print(f"Loaded {len(nx_graphs[phase])} {phase} graphs")

#         pp_dir = os.path.join(base_directory, subdir, "pandapower_networks")
#         for fn in os.listdir(pp_dir) if os.path.isdir(pp_dir) else []:
#             if fn.endswith(".json"):
#                 base = fn[:-5]
#                 gid = base[:-len(suffix)] if base.endswith(suffix) else base
#                 s = open(os.path.join(pp_dir, fn)).read()
#                 try:
#                     # pandapower ≥2.4
#                     pp_networks[phase][gid] = pp.from_json_string(s)
#                 except:
#                     pp_networks[phase][gid] = json.loads(s)
#         print(f"Loaded {len(pp_networks[phase])} {phase} graphs")

#     #return nx_graphs, pp_networks, features
#     return pp_networks 


def load_pp_networks(base_directory):
    """
    Load pandapower nets from:
      base/original/…
      base/post_MST/…
      base/optimized/…
    Correctly strips suffixes so that all phases share the same graph_id.
    """
    phases = {
        "original": {
            "subdir": "original",
            "suffix": ""                 # no suffix on original JSONs
        },
        "post_mst": {
            "subdir": "post_MST",
            "suffix": "_post_mst"        # lowercase matches filenames
        },
        "optimization": {
            "subdir": "optimized",
            "suffix": "_optimized"
        }
    }
    nets = {phase: {} for phase in phases}
    for phase, info in phases.items():
        subdir = info["subdir"]
        suffix = info["suffix"]
        folder = os.path.join(base_directory, subdir, "pandapower_networks")
        if not os.path.isdir(folder):
            logger.warning("No folder for phase '%s': %s", phase, folder)
            continue
        for fn in os.listdir(folder):
            if not fn.endswith(".json"):
                continue
            base = fn[:-5]
            # strip only the exact suffix
            if suffix and base.endswith(suffix):
                gid = base[:-len(suffix)]
            else:
                gid = base
            raw = open(os.path.join(folder, fn)).read()
            try:
                nets[phase][gid] = pp.from_json_string(raw)
            except Exception:
                raw_dict = json.loads(raw)
                nets[phase][gid] = from_json_dict(raw_dict)
        logger.info("Loaded %d pandapower nets for phase '%s'", len(nets[phase]), phase)
    return nets

# def create_pyg_data_from_nx(nx_graph, pp_network, loader_type=DataloaderType.DEFAULT,
#                             use_fallback_features=False, fallback_features=None):
#     # Validate that the graph has all required node attributes
#     if hasattr(pp_network, "bus"):
#         pp.runpp(pp_network)
#         for node in nx_graph.nodes():
#             try:
#                 res = pp_network.res_bus.loc[node]
#             except KeyError:
#                 # fallback to row-by-position
#                 res = pp_network.res_bus.iloc[int(node)]
#             nx_graph.nodes[node].update({
#                 "p":     res.p_mw,
#                 "q":     res.q_mvar,
#                 "v":     res.vm_pu,
#                 "theta": res.va_degree,
#             })
    
#     for node in nx_graph.nodes():
#         for attr in ["p", "q", "v", "theta"]:
#             if attr not in nx_graph.nodes[node]:
#                 if use_fallback_features and fallback_features and "node_features" in fallback_features:
#                     node_feats = fallback_features["node_features"]
#                     if node in node_feats and attr in node_feats[node]:
#                         nx_graph.nodes[node][attr] = node_feats[node][attr]
#                     else:
#                         raise ValueError(f"Node {node} missing attribute '{attr}' and not found in fallback features")
#                 else:
#                     raise ValueError(f"Node {node} missing required attribute '{attr}'")

#     # Validate that the graph has all required edge attributes
#     for u, v in nx_graph.edges():
#         for attr in ["R", "X", "switch_state"]:
#             if attr not in nx_graph[u][v]:
#                 if use_fallback_features and fallback_features and "edge_features" in fallback_features:
#                     edge_feats = fallback_features["edge_features"]
#                     if (u, v) in edge_feats and attr in edge_feats[(u, v)]:
#                         nx_graph[u][v][attr] = edge_feats[(u, v)][attr]
#                     elif (v, u) in edge_feats and attr in edge_feats[(v, u)]:
#                         nx_graph[u][v][attr] = edge_feats[(v, u)][attr]
#                     else:
#                         raise ValueError(f"Edge {u}-{v} missing attribute '{attr}' and not found in fallback features")
#                 else:
#                     raise ValueError(f"Edge {u}-{v} missing required attribute '{attr}'")

#             if attr in ["R", "X"] and nx_graph[u][v][attr] == 0:
#                 raise ValueError(f"Edge {u}-{v} has zero {attr} value which will cause division by zero")

#     # Create the edge_index tensor
#     edges = list(nx_graph.edges())
#     if not edges:
#         raise ValueError("Graph has no edges")
#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

#     # Extract edge attributes
#     edge_attrs = []
#     for u, v in edges:
#         edge_attrs.append([
#             nx_graph[u][v]["R"],
#             nx_graph[u][v]["X"],
#             nx_graph[u][v]["switch_state"]
#         ])
#     edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

#     # print("before loading:", pp_network[:100])
#     # print(type(pp_network))
#     # # Ensure pp_network is a dictionary before calling from_json_dict
#     # if isinstance(pp_network, str):
#     #     pp_network = json.loads(pp_network)
#     # pp_network_loaded = from_json_dict(pp_network)
#     # print("after loading:", str(pp_network_loaded)[:100])
#     # print(pp_network_loaded.line.iloc[0])

#     # line_currents = torch.tensor(pp_network_loaded.res_line.loading_percent.values, dtype=torch.float)
#     # edge_attr = torch.cat([edge_attr, line_currents.unsqueeze(1)], dim=1)

#     # Extract node features
#     num_nodes = nx_graph.number_of_nodes()
#     node_features = []
#     nodes = list(nx_graph.nodes())
#     if set(nodes) != set(range(len(nodes))):
#         raise ValueError("Graph nodes must be consecutive integers starting from 0")
#     for node_idx in range(num_nodes):
#         node_data = nx_graph.nodes[node_idx]
#         node_features.append([
#             node_data["p"],
#             node_data["q"],
#             node_data["v"],
#             node_data["theta"]
#         ])
#     x = torch.tensor(node_features, dtype=torch.float)

#     # Create the PyG Data object
#     data = Data(
#         x=x,
#         edge_index=edge_index,
#         edge_attr=edge_attr,
#         num_nodes=num_nodes
#     )

#     # Create our custom DNRDataset with the data
#     custom_data = DNRDataset(
#         x=data.x, 
#         edge_index=data.edge_index,
#         edge_attr=data.edge_attr,
#         num_nodes=data.num_nodes
#     )

#     # Add edge_y (switch state is the 3rd column - index 2)
#     switch_state_column = 2
#     custom_data.edge_y = edge_attr[:, switch_state_column].float()

#     # Add matrices based on loader type
#     if loader_type != DataloaderType.DEFAULT:
#         conductance_matrix = calculate_conductance_matrix(nx_graph)
#         coalesced = conductance_matrix.coalesce()
#         custom_data.conductance_matrix_index = coalesced.indices()
#         custom_data.conductance_matrix_values = coalesced.values()

#         adjacency_matrix = calculate_adjacency_matrix(nx_graph)
#         custom_data.adjacency_matrix_index = adjacency_matrix.coalesce().indices()
#         custom_data.adjacency_matrix_values = adjacency_matrix.coalesce().values()

#         switch_matrix = calculate_switch_matrix(nx_graph)
#         custom_data.switch_matrix_index = switch_matrix.coalesce().indices()
#         custom_data.switch_matrix_values = switch_matrix.coalesce().values()

#     # Add PINN-specific matrices
#     if loader_type == DataloaderType.PINN:
#         laplacian_matrix = calculate_laplacian_matrix(nx_graph)
#         custom_data.laplacian_matrix_index = laplacian_matrix.coalesce().indices()
#         custom_data.laplacian_matrix_values = laplacian_matrix.coalesce().values()

#         admittance_matrix = calculate_admittance_matrix(nx_graph)
#         custom_data.admittance_matrix_index = admittance_matrix.coalesce().indices()
#         custom_data.admittance_matrix_values = admittance_matrix.coalesce().values()

#     return custom_data
    
# def create_pyg_dataset_old(base_directory, loader_type=DataloaderType.DEFAULT, use_fallback_features=False):
#     nx_graphs, pp_networks, features = load_graph_data(base_directory)
    
#     data_list = []
#     successful_conversions = 0
#     failed_conversions = 0
    
#     for graph_name in nx_graphs.keys():
#         print(f"\n--- Processing graph: {graph_name} ---")
#         nx_graph = nx_graphs[graph_name]
        
#         # Get features as fallback only if requested
#         fallback_features = features.get(graph_name, None) if use_fallback_features else None
        
#         try:
#             # Create PyG data directly from the NetworkX graph, let errors propagate
#             data = create_pyg_data_from_nx(
#                 nx_graph, 
#                 pp_networks[graph_name],
#                 loader_type, 
#                 use_fallback_features=use_fallback_features,
#                 fallback_features=fallback_features
#             )
            
#             data_list.append(data)
#             successful_conversions += 1
#             print(f"Successfully converted graph: {graph_name}")
                
#         except Exception as e:
#             failed_conversions += 1
#             print(f"Error creating PyG data for {graph_name}: {e}")
#             # Let the exception propagate if this is a critical error
#             if "missing required attribute" in str(e) or "zero" in str(e):
#                 raise  # Re-raise important errors
    
#     print(f"\nCreated {len(data_list)} PyG data objects")
#     print(f"Successful conversions: {successful_conversions}")
#     print(f"Failed conversions: {failed_conversions}")
    
#     return data_list
# def create_pyg_dataset(base_directory, loader_type=DataloaderType.DEFAULT, use_fallback_features=False):
#     # new loader: returns dicts keyed by phase → {graph_id: obj}
#     nx_all, pp_all, feat_all = load_graph_data(base_directory)
#     data_list = []
 
#     # loop phases original → post_mst → optimization
#     for phase in ("original", "post_mst", "optimization"):
#         for graph_name, nx_graph in nx_all[phase].items():
#             nx_opt = nx_all["optimization"].get(graph_name)
#             if nx_opt is None:
#                 logger.warning("No optimized graph for %s, skipping y-labels", graph_name)
#                 continue
#             logger.info("Phase %s — Processing graph: %s", phase, graph_name)
#             pp_net = pp_all[phase].get(graph_name)
#             fallback = (feat_all[phase].get(graph_name)
#                         if use_fallback_features else None)
#             try:
#                 # 1) build the usual Data object from original graph
#                 data = create_pyg_data_from_nx(
#                     nx_graph,
#                     pp_net,
#                     loader_type,
#                     use_fallback_features=use_fallback_features,
#                     fallback_features=fallback
#                 )
             
#                 logger.debug("Successfully converted graph: %s", graph_name, phase)
#                 # 2) compute ground-truth switch states on each original edge
#                 #    (1 if that edge still exists in nx_opt, else 0)
#                 opt_edges = {tuple(sorted(e)) for e in nx_opt.edges()}
#                 # data.edge_index is 2×E; transpose to list of (u,v)
#                 uv = data.edge_index.t().tolist()
#                 y = torch.tensor([1.0 if tuple(sorted((u, v))) in opt_edges else 0.0
#                                 for u, v in uv],
#                                 dtype=torch.float)

#                 data.y = y
#                 data_list.append(data)
#             except Exception as e:
#                 logger.error("Error creating PyG data for %s: %s", graph_name, e)
#                 # Let the exception propagate if this is a critical error
#                 if "missing required attribute" in str(e) or "zero" in str(e):
#                     raise
#     print(f"Created {len(data_list)} Data objects for training")
#     print(f"Successful conversions: {len(data_list)}")
#     print(f"Failed conversions: {len(nx_all[phase]) - len(data_list)}")

#     return data_list


def create_pyg_from_pp(pp_net_raw, loader_type=DataloaderType.DEFAULT):
    """
    Accepts either a pandapower Net, a JSON string, or a dict.
    Converts to a Net if needed, then runs PF and extracts features.
    """
    # --- ensure we have a Net object ---
    if isinstance(pp_net_raw, str):
        # raw JSON string
        pp_net = pp.from_json_string(pp_net_raw)
    elif isinstance(pp_net_raw, dict):
        # loaded JSON dict
        pp_net = from_json_dict(pp_net_raw)
    else:
        # already a Net
        pp_net = pp_net_raw

    # --- run power flow ---
    pp.runpp(pp_net)

    # --- node features ---
    bus_res = pp_net.res_bus
    x = torch.tensor(
        np.stack([
            bus_res.p_mw.values,
            bus_res.q_mvar.values,
            bus_res.vm_pu.values,
            bus_res.va_degree.values
        ], axis=1),
        dtype=torch.float
    )

    # --- edge list & features ---
    lines = pp_net.line
    from_b = lines.from_bus.values.astype(int)
    to_b   = lines.to_bus.values.astype(int)
    edge_index = torch.tensor([from_b, to_b], dtype=torch.long)

    R = lines.r_ohm_per_km.values
    X = lines.x_ohm_per_km.values
    switches = pp_net.switch
    sw_map = {(int(r.bus), int(r.element)): int(r.closed) for _, r in switches.iterrows()}
    switch_state = [sw_map.get((u, v), 0) for u, v in zip(from_b, to_b)]
    edge_attr = torch.tensor(np.stack([R, X, switch_state], axis=1), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=x.size(0)
    )
    return data

def create_pyg_dataset_simple(
    base_directory,
    loader_type: DataloaderType = DataloaderType.DEFAULT,
    feature_phase_prob: float = 0.5,
    seed: int = None
):
    pp_all = load_pp_networks(base_directory)
    if seed is not None:
        random.seed(seed)

    data_list = []
    for gid, net_orig in pp_all["original"].items():
        net_opt = pp_all["optimization"].get(gid)
        if net_opt is None:
            logger.warning(f"No optimized net for {gid}, skipping")
            continue

        # 1) pick original vs post_mst for X-features
        phase = "post_mst" if random.random() < feature_phase_prob else "original"
        data_x = create_pyg_from_pp(pp_all[phase][gid], loader_type)

        # 2) build y-labels from optimized net
        data_y = create_pyg_from_pp(net_opt, loader_type)
        data_x.edge_y = data_y.edge_attr[:, 2]    # switch_state
        data_x.node_y_voltage = data_y.x[:, 2]    # vm_pu

        data_list.append(data_x)

    logger.info(f"Built {len(data_list)} Data objects (simple loader)")
    return data_list

def create_dynamic_loader(dataset, max_nodes=1000, max_edges=5000, shuffle=True, **kwargs):
    class DynamicBatchSampler(torch.utils.data.Sampler):
        def __init__(self, dataset, max_nodes, max_edges, shuffle):
            self.dataset = dataset
            self.max_nodes = max_nodes
            self.max_edges = max_edges
            self.shuffle = shuffle
            self.indices = list(range(len(dataset)))
        
            self.graph_sizes = [(d.num_nodes, d.num_edges) for d in dataset]
            logger.info("Sampler initialized: %d graphs; node_limit=%d, edge_limit=%d",
                        len(self.indices), max_nodes, max_edges)

        def __iter__(self):
            # Get indices of all graphs
            indices = self.indices.copy()
            
            # Shuffle if required
            if self.shuffle:
                torch.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
                torch.randperm(len(indices), out=torch.tensor(indices))
                indices = torch.randperm(len(indices)).tolist()
                logger.debug("Shuffled indices: %s", indices)
            
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
                    logger.debug("Yielding batch of %d graphs (nodes=%d, edges=%d)", len(batch), nodes, edges)
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
                logger.debug("Yielding final batch of %d graphs", len(batch))
                yield current_batch
                
        def __len__(self):
            # This is an estimate since actual number of batches depends on graph sizes
            if not self.graph_sizes:
                return 0
            total_nodes = sum(nodes for nodes, _ in self.graph_sizes)
            total_edges = sum(edges for _, edges in self.graph_sizes)
            est = max(1, min(total_nodes // self.max_nodes + 1, total_edges // self.max_edges + 1))
            logger.debug("Sampler __len__ estimate: %d", est)
            return est
    
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
    dataset = create_pyg_dataset_simple(base_directory, loader_type)
    if secondary_directory:
        print("==================================================","\n start loading secondary data")
        val_real_set = create_pyg_dataset_simple(os.path.join(secondary_directory, "validation"), loader_type)
        test_set = create_pyg_dataset_simple(os.path.join(secondary_directory, "test"), loader_type)

    
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
    log_level = os.getenv("PYTHON_LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)


    import argparse
    
    parser = argparse.ArgumentParser(description="Create data loaders for power network data")
    parser.add_argument("--base_dir", type=str,default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32/test", help="Base directory containing the train/validation folders")
    parser.add_argument("--secondary_dir", type=str, #default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_data_set_test",
                        help="Secondary directory containing the test/validation folders")
    parser.add_argument("--loader_type", type=str, default="default", 
                        choices=["default", "graphyr", "pinn",],
                        help="Type of dataloader to create")
    parser.add_argument("--batching_type", type=str, default="dynamic",
                        choices =["standard", "dynamic"])
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_nodes", type=int, default=1000, help="Maximum number of nodes in a batch (for dynamic batching)")
    parser.add_argument("--max_edges", type=int, default=5000, help="Maximum number of edges in a batch (for dynamic batching)")
    parser.add_argument("--train_ratio", type=float, default=0.85, help="Ratio of training set")
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
    
    logger.info("Data loaders created successfully.")