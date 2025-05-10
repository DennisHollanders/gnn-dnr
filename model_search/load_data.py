import os
import json
import pickle as pkl
from enum import Enum
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader, NeighborLoader 
import networkx as nx
import numpy as np
import pandapower as pp
from preprocess_data import *

from pandapower import from_json, from_json_dict
import logging 
import sys
import random
import matplotlib.pyplot as plt
import math
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Add necessary source paths
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from electrify_subgraph import extract_node_features, extract_edge_features

CACHE_ROOT_DIR = Path("data/cached_datasets")

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
            "suffix": ""                
        },
        "post_mst": {
            "subdir": "post_MST",
            "suffix": "_post_mst"       
        },
        "optimization": {
            "subdir": "optimized",
            "suffix": "_optimized"
        }
    }
    nets = {phase: {} for phase in phases}

    nets: dict[str, dict[str, pp.pandapowerNet]] = {p: {} for p in phases}

    for phase, info in phases.items():
        folder = os.path.join(base_directory,
                              info["subdir"],
                              "pandapower_networks")
        suffix = info["suffix"]
        if not os.path.isdir(folder):
            logger.warning("No folder for %s phase: %s", phase, folder)
            continue

        logger.info("Loading nets for phase '%s'…", phase)
        for fn in tqdm(os.listdir(folder), desc=f"{phase} nets", unit="file"):
            if not fn.endswith(".json"):
                continue

            gid = fn[:-5]
            if suffix and gid.endswith(suffix):
                gid = gid[:-len(suffix)]
            path = os.path.join(folder, fn)

            # ── step 1 – normal reader ───────────────────────────────
            try:
                net = pp.from_json(path)
            except Exception:
                # ── step 2 – double-encoded / exotic cases ────────────
                try:
                    with open(path) as f:
                        raw = f.read()

                    # case a) JSON string encoded twice
                    if raw.startswith('"') and raw.endswith('"'):
                        raw = json.loads(raw)

                    # try Pandapower again
                    try:
                        net = pp.from_json_string(raw)
                    except Exception:
                        # last resort: raw → dict → Net
                        net = from_json_dict(json.loads(raw))
                except Exception as e:
                    logger.error("Failed loading %s: %s", path, e)
                    continue   # skip this file

            if net.bus.empty:
                logger.warning("Skipping %s – empty bus table", gid)
                continue

            nets[phase][gid] = net

        logger.info("Loaded %d nets for phase '%s'", len(nets[phase]), phase)

    return nets


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

    # ── build a dense, contiguous bus-id → row lookup ─────────────
    bus_ids = pp_net.bus.index.to_numpy(dtype=int)           # original IDs
    id2row  = {bid: i for i, bid in enumerate(bus_ids)}      # 0 … n-1
    n_nodes = len(bus_ids)

    # --- node features ---
    bus_res = pp_net.res_bus.loc[bus_ids]     
    x = torch.tensor(np.vstack([
        bus_res.p_mw.values,
        bus_res.q_mvar.values,
        bus_res.vm_pu.values,
        bus_res.va_degree.values
    ]).T, dtype=torch.float)

    # --- edge list & features ---
    lines = pp_net.line
    n_lines = len(lines)

    # bus re-indexing stays exactly as before
    from_b = [id2row[int(b)] for b in lines.from_bus.values]
    to_b   = [id2row[int(b)] for b in lines.to_bus.values]
    
    #edge_index = torch.tensor([from_b, to_b], dtype=torch.long)
    edge_index = torch.as_tensor( np.vstack((from_b, to_b)), dtype=torch.long, device="cpu"
        )  
    #edge_index = torch.as_tensor(np.vstack((from_b, to_b)), dtype=torch.long)
    switch_state = np.ones(n_lines, dtype=int)         # default: closed

    for _, sw in pp_net.switch.iterrows():
        if sw.et == "l":                               # only line-switches
            line_idx   = int(sw.element)               # idx in  pp_net.line
            switch_state[line_idx] = int(sw.closed)    # 1 = closed, 0 = open
    R = lines.r_ohm_per_km.values
    X = lines.x_ohm_per_km.values

    # switches = pp_net.switch
    # sw_map = {(id2row[int(r.bus)], id2row[int(r.element)]): int(r.closed)
    #           for _, r in switches.iterrows() if r.et == "l"}
    edge_attr = torch.tensor(np.vstack([R, X, switch_state]).T, dtype=torch.float)
    #switch_state = [sw_map.get((u, v), 0) for u, v in zip(from_b, to_b)]
    
    #edge_attr = torch.tensor(np.vstack([R, X, switch_state]).T, dtype=torch.float)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=x.size(0)
    )

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
    for gid, net_orig in tqdm(pp_all["original"].items(), desc= f"Creating pyg data from {base_directory}"):
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
            #logger.info("Sampler initialized: %d graphs; node_limit=%d, edge_limit=%d",
            #            len(self.indices), max_nodes, max_edges)

        def __iter__(self):
            # Get indices of all graphs
            indices = self.indices.copy()
            
            # Shuffle if required
            if self.shuffle:
                torch.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
                indices = torch.randperm(len(indices)).tolist()
                #logger.debug("Shuffled indices: %s", indices)
            
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
                    #logger.debug("Yielding batch of %d graphs (nodes=%d, edges=%d)",
                        # len(current_batch),           # <-- was len(batch)
                        # current_nodes,
                        # current_edges)
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
                # logger.debug(
                #     "Yielding final batch of %d graphs",
                #     len(current_batch)        # <-- was len(batch)
                # )
                yield current_batch
                
        def __len__(self):
            # This is an estimate since actual number of batches depends on graph sizes
            if not self.graph_sizes:
                return 0
            total_nodes = sum(nodes for nodes, _ in self.graph_sizes)
            total_edges = sum(edges for _, edges in self.graph_sizes)
            est = max(1, min(total_nodes // self.max_nodes + 1, total_edges // self.max_edges + 1))
            #logger.debug("Sampler __len__ estimate: %d", est)
            return est
    
    # Create the batch sampler
    batch_sampler = DynamicBatchSampler(dataset, max_nodes, max_edges, shuffle)
    
    # Create DataLoader with the custom batch sampler
    return DataLoader(
        dataset, 
        batch_sampler=batch_sampler, 
        collate_fn= lambda batch: Batch.from_data_list(batch),
        **kwargs
    )
def create_neighbor_loaders(dataset, num_neighbors=[15, 10], batch_size=1024, **kwargs):
    return NeighborLoader(
        dataset,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )


def _get_cache_path_suffix(original_path_str: str) -> Path:
    """Helper function to determine the suffix for the cache path."""
    original_path = Path(original_path_str)
    if original_path.parts and original_path.parts[0].lower() == "data":
        return Path(*original_path.parts[1:])
    return original_path

def _load_or_create_dataset(
    dataset_name: str,
    input_data_path_str: str,
    dataset_cache_file_path: Path, # Accepts the full path to the cache file
    loader_type: DataloaderType,
    seed: int
):
    """Handles loading a single dataset from cache or creating and caching it."""
    dataset_cache_dir = dataset_cache_file_path.parent # Derive dir from full file path
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    
    loaded_dataset = []

    if os.path.exists(dataset_cache_file_path):
        logger.info(f"Loading {dataset_name} dataset from cache: {dataset_cache_file_path}")
        try:
            loaded_dataset = torch.load(dataset_cache_file_path, weights_only=False)
            if not isinstance(loaded_dataset, list):
                loaded_dataset = list(loaded_dataset)
            logger.info(f"{dataset_name} dataset loaded from cache ({len(loaded_dataset)} samples).")
        except Exception as e:
            logger.error(f"Failed to load {dataset_name} dataset from cache: {e}. Regenerating.", exc_info=True)
            loaded_dataset = [] # Ensure it's an empty list to trigger generation
    
    if not loaded_dataset: # If cache miss or loading failed
        logger.info(f"Generating {dataset_name} dataset for: {input_data_path_str}")
        created_data = create_pyg_dataset_simple(input_data_path_str, loader_type, seed=seed)
        loaded_dataset = created_data if created_data is not None else []

        if not loaded_dataset and input_data_path_str: # Check if still empty after attempting generation
            logger.error(f"Failed to generate {dataset_name} dataset for {input_data_path_str}.")
            return [] 
        
        try:
            logger.info(f"Saving generated {dataset_name} dataset to cache: {dataset_cache_file_path}")
            torch.save(loaded_dataset, dataset_cache_file_path)
            logger.info(f"{dataset_name} dataset saved to cache.")
        except Exception as e:
            logger.error(f"Failed to save {dataset_name} dataset to cache: {e}", exc_info=True)
            # Proceed with in-memory data even if saving fails

    return loaded_dataset


def create_data_loaders(
    base_directory: str,
    secondary_directory: str = None,
    loader_type: DataloaderType = DataloaderType.DEFAULT,
    batch_size: int = 32,
    max_nodes: int = 1000,
    max_edges: int = 5000,
    transform=None,
    train_ratio: float = 0.85,
    seed: int = 0,
    batching_type: str = "standard",
    num_workers: int = 0,
):
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    CACHE_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Synthetic Dataset (from base_directory, for train/val_synthetic split) ---
    # The cache directory key is derived from the parent of the specific data folder (e.g., parent of 'test')
    # base_directory is like 'data/SET_NAME/test'
    # We want the cache key for the set, which is 'SET_NAME'
    base_set_name_path_str = str(Path(base_directory).parent) 
    base_set_cache_key = _get_cache_path_suffix(base_set_name_path_str)
    
    synthetic_cache_dir = CACHE_ROOT_DIR / base_set_cache_key
    synthetic_cache_filename = f"{Path(base_directory).name}.pt" # e.g., "test.pt"
    synthetic_cache_file_path = synthetic_cache_dir / synthetic_cache_filename

    synthetic_dataset = _load_or_create_dataset(
        dataset_name="synthetic (from base_dir)",
        input_data_path_str=base_directory,
        dataset_cache_file_path=synthetic_cache_file_path,
        loader_type=loader_type,
        seed=seed
    )
    if not synthetic_dataset and base_directory:
        logger.error("Base dataset (for synthetic) generation failed. Cannot proceed.")
        return None, None, None, None

    # --- Real Validation Set (from secondary_directory/validation) ---
    val_real_set = []
    if secondary_directory:
        # secondary_directory is like 'data/SET_NAME'
        secondary_set_cache_key = _get_cache_path_suffix(secondary_directory)
        
        val_real_input_path_str = str(Path(secondary_directory) / "validation")
        val_real_cache_dir = CACHE_ROOT_DIR / secondary_set_cache_key
        val_real_cache_filename = "validation.pt"
        val_real_cache_file_path = val_real_cache_dir / val_real_cache_filename

        val_real_set = _load_or_create_dataset(
            dataset_name="real validation",
            input_data_path_str=val_real_input_path_str,
            dataset_cache_file_path=val_real_cache_file_path,
            loader_type=loader_type,
            seed=seed
        )
    else:
        logger.info("No secondary directory provided for real validation set.")

    test_set = [] 
    if secondary_directory:
        # secondary_directory is like 'data/SET_NAME_SECONDARY'
        secondary_set_cache_key = _get_cache_path_suffix(secondary_directory)
        
        test_input_path_str = str(Path(secondary_directory) / "test")
        test_cache_dir = CACHE_ROOT_DIR / secondary_set_cache_key # Same cache dir as val_real_set
        test_cache_filename = "test.pt" # Specific filename for this dataset
        test_cache_file_path = test_cache_dir / test_cache_filename

        test_set = _load_or_create_dataset(
            dataset_name="test set (from secondary_dir/test)",
            input_data_path_str=test_input_path_str,
            dataset_cache_file_path=test_cache_file_path,
            loader_type=loader_type,
            seed=seed
        )
        if not test_set: # Check if test set loading/generation failed
            logger.warning(f"Test set from {test_input_path_str} could not be loaded/generated.")
    else:
        logger.info("No secondary directory provided, so no dedicated test set will be loaded from it for the 'test_set' variable.")
        # test_set remains [] if secondary_directory is None

    # --- Perform Train/Validation Split on the Base Dataset ---
    train_set = []
    val_synthetic_set = []
    if synthetic_dataset:
        logger.info("Performing train/validation split on base dataset (synthetic_dataset).")
        train_size = int(train_ratio * len(synthetic_dataset))
        
        if len(synthetic_dataset) < 2:
            logger.warning("Base dataset (synthetic_dataset) too small for split. Adjusting.")
            if train_ratio > 0 or len(synthetic_dataset) == 0 : # if train_ratio is 0, it all goes to val
                train_set = list(synthetic_dataset)
                val_synthetic_set = []
            else: # train_ratio is 0 and dataset has 1 element
                train_set = []
                val_synthetic_set = list(synthetic_dataset)
        elif train_size == len(synthetic_dataset) and train_size > 0 :
            train_set = list(synthetic_dataset)
            val_synthetic_set = []
            logger.warning("Train ratio resulted in full dataset for training, no synthetic validation.")
        elif train_size == 0 and len(synthetic_dataset) > 0:
            train_set = []
            val_synthetic_set = list(synthetic_dataset)
            logger.warning("Train ratio resulted in zero for training, using all for synthetic validation.")
        else: # Regular split for len(synthetic_dataset) >= 2
            train_set, val_synthetic_set = torch.utils.data.random_split(
                synthetic_dataset, [train_size, len(synthetic_dataset) - train_size], generator=generator
            )
            train_set = list(train_set)
            val_synthetic_set = list(val_synthetic_set)
        logger.info(f"Split base dataset: Train ({len(train_set)}), Synthetic Val ({len(val_synthetic_set)})")
    else:
        logger.warning("Base dataset (synthetic_dataset) is empty. Train and synthetic validation sets will be empty.")
        
    # --- Apply transform to all datasets ---
    if transform:
        logger.info("Applying transform to datasets.")
        if train_set: train_set = [transform(data) for data in train_set]
        if val_synthetic_set: val_synthetic_set = [transform(data) for data in val_synthetic_set]
        if val_real_set: val_real_set = [transform(data) for data in val_real_set]
        if test_set: test_set = [transform(data) for data in test_set]
        logger.info("Transform application finished.")
    else:
        logger.info("No transform specified.")

    # --- Create DataLoaders ---
    logger.info("Creating data loaders.")
    train_loader, val_synthetic_loader, val_real_loader, test_loader = None, None, None, None

    def _create_loader(dataset, is_train_loader): # Renamed for clarity
        if not dataset: return None
        if batching_type == "dynamic":
            return create_dynamic_loader(dataset, max_nodes=max_nodes, max_edges=max_edges, shuffle=is_train_loader, num_workers=num_workers)
        elif batching_type == "standard":
            return DataLoader(dataset, batch_size=batch_size, shuffle=is_train_loader, num_workers=num_workers)
        # Add other batching types if needed
        return None

    train_loader = _create_loader(train_set, True)
    val_synthetic_loader = _create_loader(val_synthetic_set, False)
    val_real_loader = _create_loader(val_real_set, False)
    test_loader = _create_loader(test_set, False)
    
    if batching_type == "neighbor": # Specific warning if this type is chosen
         logger.error("NeighborLoader batching_type not fully implemented for list[Data] in this setup. Loaders will be None if this was the intended type for all.")

    print(f"\nCreated data loaders with:")
    print(f"  Training samples: {len(train_set)}")
    print(f"  Synthetic validation samples: {len(val_synthetic_set)}")
    print(f"  Real validation samples: {len(val_real_set)}")
    print(f"  Test samples: {len(test_set)}") # This is now from base_directory

    return train_loader, val_synthetic_loader, val_real_loader, test_loader



def _collect_batch_stats(loader):
    """Return list[(n_nodes, n_edges)] for *all* batches in loader."""
    stats = []
    for batch in loader:
        if hasattr(batch, "num_graphs"):   # PyG Batch or NeighborSampler output
            stats.append((batch.num_nodes, batch.num_edges))
        elif isinstance(batch, list):      # DynamicBatchSampler → list[Data]
            n = sum(d.num_nodes for d in batch)
            e = sum(d.num_edges for d in batch)
            stats.append((n, e))
        else:                              # Fallback single‐graph batch
            stats.append((batch.num_nodes, batch.num_edges))
    return stats

def _plot_hist(vals, title, fname):
    plt.figure()
    plt.hist(vals, bins=min(30, int(math.sqrt(len(vals))) + 2))
    plt.title(title); plt.xlabel("count"); plt.ylabel("#batches")
    plt.tight_layout(); plt.savefig(fname); plt.close()


if __name__ == "__main__":
    # logging.basicConfig(
    # level=logging.INFO,
    # format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    # datefmt="%Y-%m-%d %H:%M:%S",
    # )
    # logger = logging.getLogger(__name__)
    # log_level = os.getenv("PYTHON_LOG_LEVEL", "INFO").upper()
    # logger.setLevel(log_level)
    # # # ─── configure root logger ─────────────────────────────────────────────────────

    import argparse
    
    parser = argparse.ArgumentParser(description="Create data loaders for power network data")
    parser.add_argument("--base_dir", type=str,
                        #default=r"data\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32/test",
                        
                        default = r"data\test_val_real__range-30-230_nTest-1000_nVal-1000_2842025_1\test",
                        help="Base directory containing the train/validation folders")
    parser.add_argument("--secondary_dir", type=str,
                         #default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_data_set_test",
                        default=r"data\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32",
                        help="Secondary directory containing the test/validation folders")
    parser.add_argument("--loader_type", type=str, default="default", 
                        choices=["default", "graphyr", "pinn",],
                        help="Type of dataloader to create")
    parser.add_argument("--batching_type", type=str, default="dynamic",
                        choices =["standard", "dynamic", "neighbor"],)
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
    
    print("base_dir:", args.base_dir)
    print("secondary_dir:", args.secondary_dir)
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
        batching_type = args.batching_type,
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
