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
    # for phase, info in phases.items():
    #     subdir = info["subdir"]
    #     suffix = info["suffix"]
    #     folder = os.path.join(base_directory, subdir, "pandapower_networks")
    #     if not os.path.isdir(folder):
    #         logger.warning("No folder for phase '%s': %s", phase, folder)
    #         continue
    #     for fn in os.listdir(folder):
    #         if not fn.endswith(".json"):
    #             continue
    #         base = fn[:-5]
    #         # strip only the exact suffix
    #         if suffix and base.endswith(suffix):
    #             gid = base[:-len(suffix)]
    #         else:
    #             gid = base
    #         raw = open(os.path.join(folder, fn)).read()
    #         try:
    #             nets[phase][gid] = pp.from_json_string(raw)
    #         except Exception:
    #             raw_dict = json.loads(raw)
    #             nets[phase][gid] = from_json_dict(raw_dict)
    #     logger.info("Loaded %d pandapower nets for phase '%s'", len(nets[phase]), phase)
    # return nets
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
            print(type(path))
            try:
                net = pp.from_json(path)
            except Exception:
                # ── step 2 – double-encoded / exotic cases ────────────
                try:
                    with open(path) as f:
                        raw = f.read()
                        print(type(raw))

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
                    logger.debug("Yielding batch of %d graphs (nodes=%d, edges=%d)",
                        len(current_batch),           # <-- was len(batch)
                        current_nodes,
                        current_edges)
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
                logger.debug(
                    "Yielding final batch of %d graphs",
                    len(current_batch)        # <-- was len(batch)
                )
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

def create_data_loaders(
    base_directory,
    secondary_directory=None,
    loader_type: DataloaderType = DataloaderType.DEFAULT,
    batch_size=32, max_nodes=1000, max_edges=5000,
    transform=None, train_ratio=0.85, seed=0,
    batching_type="standard", num_workers=1,
):
    """
    Creates data loaders for power network data with caching datasets.
    Generates and saves datasets if cache is not found.
    """
    # Optional: check_directory_structure(base_directory, secondary_directory)
    # It's good practice, uncomment if desired.

    # Use a generator for reproducible splits
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed) # Ensure other random operations (like feature_phase_prob) are also reproducible
    random.seed(seed) # Seed random module used in create_pyg_dataset_simple


    # 1. Define cache directory and paths
    # Use a cache directory that incorporates relevant parameters for uniqueness
    base_dir_part = Path(base_directory).name.replace('.', '_').replace('/', '_').replace('\\', '_')
    # Handle cases where base_directory itself is a root or drive letter
    if not base_dir_part:
         base_dir_part = Path(base_directory).stem.replace('.', '_') if Path(base_directory).stem else "root"

    secondary_dir_part = ""
    if secondary_directory:
        secondary_dir_part = Path(secondary_directory).name.replace('.', '_').replace('/', '_').replace('\\', '_')
        if not secondary_dir_part:
            secondary_dir_part = Path(secondary_directory).stem.replace('.', '_') if Path(secondary_directory).stem else "root"

    # Incorporate parameters that change dataset generation/split
    cache_subdir_name_parts = [base_dir_part]
    if secondary_directory:
        cache_subdir_name_parts.append(secondary_dir_part)

    cache_subdir_name_parts.extend([
         loader_type.value,
         f"s{seed}",
         f"r{int(train_ratio*100)}"
    ])
    cache_subdir_name = "_".join(cache_subdir_name_parts)


    cache_dir = Path("data") / "cached_datasets" / cache_subdir_name
    cache_dir.mkdir(parents=True, exist_ok=True) # Ensure cache directory exists


    # Define specific file paths within the cache directory
    # Use consistent names regardless of input directory names within the cache folder
    synthetic_dataset_path = cache_dir / "synthetic_dataset.pt"
    val_real_dataset_path = cache_dir / "val_real_dataset.pt"
    test_dataset_path = cache_dir / "test_dataset.pt"


    # Initialize dataset variables - they will be populated from cache or generation
    synthetic_dataset = None # This holds the full base dataset before splitting
    train_set = None
    val_synthetic_set = None
    val_real_set = None
    test_set = None

    # 2. Check cache availability for base and secondary datasets separately
    cache_found_base = os.path.exists(synthetic_dataset_path)
    cache_found_secondary = False
    if secondary_directory:
         cache_found_secondary = os.path.exists(val_real_dataset_path) and os.path.exists(test_dataset_path)


    logger.info(f"Cache found for base data ({synthetic_dataset_path}): {cache_found_base}")
    if secondary_directory:
         logger.info(f"Cache found for secondary data ({val_real_dataset_path}, {test_dataset_path}): {cache_found_secondary}")


    # 3. Load or Generate Base Data
    if cache_found_base:
        logger.info(f"Cache found for base data at {synthetic_dataset_path}, loading dataset...")
        try:
            # **FIX:** Use the correct path and disable weights_only
            synthetic_dataset = torch.load(synthetic_dataset_path, weights_only=False)
            logger.info(f"Base dataset loaded from cache ({len(synthetic_dataset) if synthetic_dataset is not None else 0} samples).")
            if not isinstance(synthetic_dataset, list): # Ensure it's a list after loading
                 synthetic_dataset = list(synthetic_dataset)
                 logger.warning("Loaded base dataset was not a list, converted to list.")
        except Exception as e:
            logger.error(f"Failed to load base dataset from cache: {e}. Proceeding with data generation.", exc_info=True) # Log traceback
            synthetic_dataset = None # Ensure it's None to trigger generation

    if synthetic_dataset is None or not synthetic_dataset: # If cache failed, not found, or loaded empty list
        logger.info("==================================================")
        logger.info("Start loading and generating base data") # **FIX:** Corrected logging message
        synthetic_dataset = create_pyg_dataset_simple(base_directory, loader_type, seed=seed) # Pass seed for internal randomness

        if not synthetic_dataset:
            logger.error("Failed to generate base dataset. Cannot proceed.")
            # Return None for all loaders
            return None, None, None, None

        # Save the generated base dataset
        logger.info(f"Saving generated base dataset to cache at {synthetic_dataset_path}...")
        try:
            # **FIX:** Save the generated dataset
            torch.save(synthetic_dataset, synthetic_dataset_path)
            logger.info("Base dataset saved to cache.")
        except Exception as e:
            logger.error(f"Failed to save base dataset to cache: {e}", exc_info=True) # Log traceback


    # 4. Load or Generate Secondary Data
    if secondary_directory:
        if cache_found_secondary:
            logger.info(f"Cache found for secondary data at {val_real_dataset_path} and {test_dataset_path}, loading datasets...")
            try:
                # **FIX:** Use the correct paths and disable weights_only
                val_real_set = torch.load(val_real_dataset_path, weights_only=False)
                test_set = torch.load(test_dataset_path, weights_only=False)
                logger.info(f"Secondary datasets loaded from cache (Val:{len(val_real_set) if val_real_set is not None else 0}, Test:{len(test_set) if test_set is not None else 0} samples).")
                # Ensure they are lists after loading
                if not isinstance(val_real_set, list): val_real_set = list(val_real_set)
                if not isinstance(test_set, list): test_set = list(test_set)
            except Exception as e:
                 logger.error(f"Failed to load secondary datasets from cache: {e}. Proceeding with data generation.", exc_info=True) # Log traceback
                 val_real_set = None # Ensure None to trigger generation
                 test_set = None
        # else: # Cache not found - handled by the generation block below


        if val_real_set is None or test_set is None or not val_real_set or not test_set: # If cache failed, not found, or loaded empty lists
            logger.info("==================================================")
            logger.info("Start loading and generating secondary data") # **FIX:** Corrected logging message
            val_real_set = create_pyg_dataset_simple(os.path.join(secondary_directory, "validation"), loader_type, seed=seed) # Pass seed? Depends if randomness matters for secondary data generation.
            test_set = create_pyg_dataset_simple(os.path.join(secondary_directory, "test"), loader_type, seed=seed) # Pass seed?

            # Ensure lists even if None was returned by create_pyg_dataset_simple
            if val_real_set is None: val_real_set = []
            if test_set is None: test_set = []

            # Save the generated secondary datasets
            logger.info(f"Saving generated secondary datasets to cache at {val_real_dataset_path} and {test_dataset_path}...")
            try:
                # **FIX:** Save the generated datasets
                torch.save(val_real_set, val_real_dataset_path)
                torch.save(test_set, test_dataset_path)
                logger.info("Secondary datasets saved to cache.")
            except Exception as e:
                logger.error(f"Failed to save secondary datasets to cache: {e}", exc_info=True) # Log traceback
    else: # No secondary_directory provided
        val_real_set = [] # Explicitly ensure empty lists
        test_set = []
        logger.info("No secondary directory provided. Real validation and test datasets are empty.")


    # 5. Perform Train/Validation Split on the Base Dataset
    # This happens *after* the base dataset is guaranteed to be loaded or generated.
    logger.info("==================================================") # **FIX:** Corrected logging message
    logger.info("Performing train/validation split on base dataset")

    # Ensure synthetic_dataset is valid before splitting
    if not synthetic_dataset or not isinstance(synthetic_dataset, list):
         logger.error("Base synthetic dataset is empty or not a list. Cannot perform split.")
         # Return None for loaders or handle appropriately
         train_set = [] # Ensure empty lists for consistency
         val_synthetic_set = []
    else:
        train_size = int(train_ratio * len(synthetic_dataset))

        # Handle edge cases for splitting small datasets
        if len(synthetic_dataset) < 2:
             logger.warning("Synthetic dataset size is less than 2. Cannot perform train/val split.")
             train_set = list(synthetic_dataset) # Use the whole dataset for training or handle as needed
             val_synthetic_set = [] # Empty val set
        elif train_size == 0:
             logger.warning("Calculated train set size is 0. Entire synthetic dataset used for validation.")
             train_set = [] # Empty train set
             val_synthetic_set = list(synthetic_dataset)
        elif train_size == len(synthetic_dataset):
             logger.warning("Calculated train set size equals dataset size. No synthetic validation set.")
             train_set = list(synthetic_dataset)
             val_synthetic_set = [] # Empty val set
        else:
            # Perform the random split
            train_set, val_synthetic_set = torch.utils.data.random_split(
                synthetic_dataset, [train_size, len(synthetic_dataset) - train_size], generator=generator
            )
            # Convert Subset objects from random_split back to lists for consistency
            train_set = list(train_set)
            val_synthetic_set = list(val_synthetic_set)

    logger.info(f"Split base dataset: Train ({len(train_set)}), Synthetic Val ({len(val_synthetic_set)})")


    # 6. Apply transform to all datasets (after loading/generating and splitting synthetic)
    logger.info("==================================================") # **FIX:** Corrected logging message
    logger.info("Start applying transform")
    if transform:
        # Check if datasets are not None and are lists before transforming
        # The split ensures train_set and val_synthetic_set are lists, but check just in case
        if train_set is not None and isinstance(train_set, list):
             logger.info(f"Applying transform to {len(train_set)} train samples.")
             train_set = [transform(data) for data in train_set]
        else: logger.debug("Train set is None or not a list (or empty), skipping transform.")

        if val_synthetic_set is not None and isinstance(val_synthetic_set, list):
             logger.info(f"Applying transform to {len(val_synthetic_set)} synthetic validation samples.")
             val_synthetic_set = [transform(data) for data in val_synthetic_set]
        else: logger.debug("Synthetic validation set is None or not a list (or empty), skipping transform.")

        # Only attempt secondary transforms if secondary_directory was originally provided AND datasets are lists
        if secondary_directory:
             if val_real_set is not None and isinstance(val_real_set, list):
                  logger.info(f"Applying transform to {len(val_real_set)} real validation samples.")
                  val_real_set = [transform(data) for data in val_real_set]
             else: logger.debug("Real validation set is None or not a list (or empty), skipping transform.")

             if test_set is not None and isinstance(test_set, list):
                  logger.info(f"Applying transform to {len(test_set)} test samples.")
                  test_set = [transform(data) for data in test_set]
             else: logger.debug("Test set is None or not a list (or empty), skipping transform.")

        logger.info("Transform application finished.")
    else:
        logger.info("No transform specified, skipping transform step.")


    # 7. Create DataLoaders from the datasets
    logger.info("==================================================") # **FIX:** Corrected logging message
    logger.info("Start creating loaders")

    # Initialize loaders to None
    train_loader = None
    val_synthetic_loader = None
    val_real_loader = None
    test_loader = None

    # Create loaders only if the corresponding dataset list is not empty
    # Dynamic and Standard Loaders expect a list of Data objects
    if batching_type == "dynamic":
        if train_set: # Check if list is not empty/None
             train_loader = create_dynamic_loader(train_set, max_nodes=max_nodes, max_edges=max_edges, shuffle=True, num_workers=num_workers)
        else: logger.warning("Train dataset is empty, train_loader is None.")

        if val_synthetic_set: # Check if list is not empty/None
             val_synthetic_loader = create_dynamic_loader(val_synthetic_set, max_nodes=max_nodes, max_edges=max_edges, shuffle=False, num_workers=num_workers)
        else: logger.warning("Synthetic validation dataset is empty, val_synthetic_loader is None.")

        if secondary_directory:
             if val_real_set: # Check if list is not empty/None
                  val_real_loader = create_dynamic_loader(val_real_set, max_nodes=max_nodes, max_edges=max_edges, shuffle=False, num_workers=num_workers)
             else: logger.warning("Real validation dataset is empty, val_real_loader is None.")

             if test_set: # Check if list is not empty/None
                  test_loader = create_dynamic_loader(test_set, max_nodes=max_nodes, max_edges=max_edges, shuffle=False, num_workers=num_workers)
             else: logger.warning("Test dataset is empty, test_loader is None.")
        else: logger.info("No secondary directory, val_real_loader and test_loader remain None.")


    elif batching_type == "neighbor":
         # **FIX:** NeighborLoader support for list[Data] is not standard. Keep as placeholder returning None.
         logger.error("NeighborLoader batching_type not fully implemented for dataset lists. Loaders will be None.")
         train_loader = None
         val_synthetic_loader = None
         val_real_loader = None
         test_loader = None

    else: # Use standard DataLoader
        if train_set: # Check if list is not empty/None
             train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        else: logger.warning("Train dataset is empty, train_loader is None.")

        if val_synthetic_set: # Check if list is not empty/None
             val_synthetic_loader = DataLoader(val_synthetic_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else: logger.warning("Synthetic validation dataset is empty, val_synthetic_loader is None.")

        if secondary_directory:
             if val_real_set: # Check if list is not empty/None
                  val_real_loader = DataLoader(val_real_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
             else: logger.warning("Real validation dataset is empty, val_real_loader is None.")

             if test_set: # Check if list is not empty/None
                  test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
             else: logger.warning("Test dataset is empty, test_loader is None.")
        else: logger.info("No secondary directory, val_real_loader and test_loader remain None.")


    # Print counts - handle potential None datasets gracefully
    # Use the variables *after* they have been assigned
    train_count = len(train_set) if train_set is not None and isinstance(train_set, list) else 0
    val_synth_count = len(val_synthetic_set) if val_synthetic_set is not None and isinstance(val_synthetic_set, list) else 0
    val_real_count = len(val_real_set) if val_real_set is not None and isinstance(val_real_set, list) else 0
    test_count = len(test_set) if test_set is not None and isinstance(test_set, list) else 0


    print(f"\nCreated data loaders with:")
    print(f"  Training samples: {train_count}")
    print(f"  Synthetic validation samples: {val_synth_count}")
    print(f"  Real validation samples: {val_real_count}")
    print(f"  Test samples: {test_count}")


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

def run_loader_diagnostics(train_loader, val_synth_loader,
                           val_real_loader=None, test_loader=None,
                           out_dir="diagnostics"):
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
    all_loaders = {
        "train": train_loader,
        "val_synth": val_synth_loader,
        "val_real": val_real_loader,
        "test": test_loader,
    }

    # ── iterate once through every loader ────────────────────────────────────
    for name, loader in all_loaders.items():
        if loader is None: continue
        stats = _collect_batch_stats(loader)
        if not stats: continue

        n_nodes, n_edges = zip(*stats)
        _plot_hist(n_nodes, f"{name}: nodes per batch", out_dir/f"{name}_nodes.png")
        _plot_hist(n_edges, f"{name}: edges per batch", out_dir/f"{name}_edges.png")

        print(f"\n[{name.upper()}]  batches={len(stats)}")
        print(f"  nodes  :  min={min(n_nodes)}  max={max(n_nodes)}  mean={sum(n_nodes)/len(stats):.1f}")
        print(f"  edges  :  min={min(n_edges)}  max={max(n_edges)}  mean={sum(n_edges)/len(stats):.1f}")

        # ── attribute checks on first batch ────────────────────────────────
        first = next(iter(loader))
        msg = f"  attrs  : " + ", ".join(sorted(first.__dict__.keys()))
        print(msg)

        assert first.edge_index.size(1) == first.edge_attr.size(0), \
            f"{name}: edge_index vs edge_attr mismatch"
        if hasattr(first, "edge_y"):
            assert first.edge_y.size(0) == first.edge_index.size(1), \
                f"{name}: edge_y length mismatch"

        for tensor_name, t in first.__dict__.items():
            if torch.is_tensor(t):
                assert torch.isfinite(t).all(), f"{name}: {tensor_name} contains NaN/Inf"

        # extra matrices for GRAPHYR / PINN
        for mat in ["conductance_matrix_index", "laplacian_matrix_index"]:
            if hasattr(first, mat):
                idx = getattr(first, mat)
                assert idx.dim() == 2 and idx.size(0) == 2, f"{name}: {mat} malformed"

    print(f"\nDiagnostic plots saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    log_level = os.getenv("PYTHON_LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)


    import argparse
    
    parser = argparse.ArgumentParser(description="Create data loaders for power network data")
    parser.add_argument("--base_dir", type=str,
                        #default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32/test", 
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

    # Run the diagnostics on the loaders
    run_loader_diagnostics(train_loader, val_synthetic_loader,
                           val_real_loader=val_real_loader, test_loader=test_loader,
                           out_dir="diagnostics")