import os
import json
import pickle as pkl
from enum import Enum
from unittest import loader
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader, NeighborLoader, DynamicBatchSampler
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

from preprocess_data import * 

CACHE_ROOT_DIR = Path("data/cached_datasets")

logger = logging.getLogger(__name__)

class DataloaderType(Enum):
    DEFAULT = "default"
    GRAPHYR = "graphyr"
    PINN = "pinn"
    CVX = "cvx"

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


def create_pyg_from_pp(pp_net_raw):
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
    # pp.runpp(pp_net)

    # ── build a dense, contiguous bus-id → row lookup ─────────────
    bus_ids = pp_net.bus.index.to_numpy(dtype=int)           # original IDs
    id2row  = {bid: i for i, bid in enumerate(bus_ids)}      # 0 … n-1)

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

        # Re-index bus connections using pandas' map for efficiency
    from_b = lines.from_bus.map(id2row).to_numpy(dtype=np.int64)
    to_b   = lines.to_bus.map(id2row).to_numpy(dtype=np.int64)
    
    edge_index = torch.from_numpy(np.vstack((from_b, to_b)))

    switch_state = np.ones(n_lines, dtype=int)
    line_switches = pp_net.switch[pp_net.switch.et == "l"]
    # build a map from line-ID → row idx in `lines`
    line_idx = lines.index.to_numpy(dtype=int)
    id2pos   = {lid: pos for pos, lid in enumerate(line_idx)}
    # extract positions for each switch’s element
    elems    = line_switches.element.to_numpy(dtype=int)
    positions = np.array([id2pos[e] for e in elems], dtype=int)
    switch_state[positions] = line_switches.closed.to_numpy(dtype=int)

    R = lines.r_ohm_per_km.to_numpy()
    X = lines.x_ohm_per_km.to_numpy()
    
    edge_attr = torch.from_numpy(np.vstack([R, X, switch_state]).T).float()


    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=x.size(0)
    )

def create_pyg_dataset(
    base_directory,
    dataset_type: str ="default",
    feature_phase_prob: float = 0.5,
    seed: int = None
):
    if dataset_type == "cvx":
        logger.info("including cvx features")

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
        phase = "post_mst" #if random.random() < feature_phase_prob else "original"
        data_x = create_pyg_from_pp(pp_all[phase][gid])

        # 2) build y-labels from optimized net
        data_y = create_pyg_from_pp(net_opt)
        data_x.edge_y = data_y.edge_attr[:, 2]    # switch_state
        data_x.node_y_voltage = data_y.x[:, 2]    # vm_pu

        if dataset_type == "cvx":
            cvx_feat =cxv_features(pp_all[phase][gid] )
            # Store CVX features as additional attributes in the Data object
            data_x.cvx_N = cvx_feat['N']
            data_x.cvx_E = cvx_feat['E']
            data_x.cvx_from_idx = cvx_feat['from_idx']
            data_x.cvx_to_idx = cvx_feat['to_idx']
            data_x.cvx_r_pu = cvx_feat['r_pu']
            data_x.cvx_x_pu = cvx_feat['x_pu']
            data_x.cvx_p_inj = cvx_feat['p_inj']
            data_x.cvx_q_inj = cvx_feat['q_inj']
            data_x.cvx_bigM_flow = cvx_feat['bigM_flow']
            data_x.cvx_bigM_v = torch.tensor(cvx_feat['bigM_v'], dtype=torch.float32)
            data_x.cvx_sub_idx = cvx_feat['sub_idx']
            data_x.cvx_y0 = cvx_feat['y0']

        # sanity checks:
        if data_x.num_nodes != data_y.num_nodes:
            raise ValueError(f"{gid}: node count mismatch X={data_x.num_nodes} vs Y={data_y.num_nodes}")
        if data_x.edge_index.size(1) != data_y.edge_index.size(1):
            raise ValueError(f"{gid}: edge count mismatch X={data_x.edge_index.size(1)} vs Y={data_y.edge_index.size(1)}")


        data_list.append(data_x)

    logger.info(f"Built {len(data_list)} Data objects (simple loader)")
    return data_list

def create_dynamic_loader(dataset, max_nodes=1000, max_edges=5000, shuffle=True, **kwargs):
    est_batches = math.ceil(sum(d.num_nodes for d in dataset) / max_nodes)
    batch_sampler = DynamicBatchSampler(dataset, max_nodes,mode="node", shuffle=shuffle, num_steps =est_batches)
    return DataLoader(
        dataset, 
        batch_sampler=batch_sampler, 
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
    dataset_type: str = "default",
    seed: int = 0
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
        created_data = create_pyg_dataset(input_data_path_str, dataset_type, seed=seed)
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
    dataset_type: str = "default",	
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
    synthetic_cache_filename = f"{Path(base_directory).name}-{dataset_type}.pt"
    synthetic_cache_file_path = synthetic_cache_dir / synthetic_cache_filename 

    synthetic_dataset = _load_or_create_dataset(
        dataset_name="synthetic (from base_dir)",
        input_data_path_str=base_directory,
        dataset_cache_file_path=synthetic_cache_file_path,
        dataset_type=dataset_type,
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
        val_real_cache_file_path = val_real_cache_dir / f"validation-{dataset_type}.pt"

        val_real_set = _load_or_create_dataset(
            dataset_name="real validation",
            input_data_path_str=val_real_input_path_str,
            dataset_cache_file_path=val_real_cache_file_path,
            dataset_type=dataset_type,
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
        test_cache_file_path = test_cache_dir / f"test-{dataset_type}.pt"

        test_set = _load_or_create_dataset(
            dataset_name="test set (from secondary_dir/test)",
            input_data_path_str=test_input_path_str,
            dataset_cache_file_path=test_cache_file_path,
            dataset_type=dataset_type,
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
        loader_kwargs = {
            'num_workers': num_workers,
            'pin_memory': False,  # No GPU, so no need to pin memory
            'persistent_workers': True if num_workers > 0 else False
        }
        if batching_type == "dynamic":
            return create_dynamic_loader(dataset, max_nodes=max_nodes, max_edges=max_edges, shuffle=is_train_loader, **loader_kwargs)
        
        elif batching_type == "standard":
            return DataLoader(dataset, batch_size=batch_size, shuffle=is_train_loader,**loader_kwargs)
        # Add other batching types if needed
        return None

    train_loader = _create_loader(train_set, True)
    val_synthetic_loader = _create_loader(val_synthetic_set, False)
    val_real_loader = _create_loader(val_real_set, False)
    test_loader = _create_loader(test_set, False)
    
    if batching_type == "neighbor": 
         logger.error("NeighborLoader batching_type not fully implemented for list[Data] in this setup. Loaders will be None if this was the intended type for all.")

    print(f"\nCreated data loaders with:")
    print(f"  Training samples: {len(train_set)}")
    print(f"  Synthetic validation samples: {len(val_synthetic_set)}")
    print(f"  Real validation samples: {len(val_real_set)}")
    print(f"  Test samples: {len(test_set)}") # This is now from base_directory

    return train_loader, val_synthetic_loader, val_real_loader, test_loader


if __name__ == "__main__":
    # logging.basicConfig(
    # level=logging.INFO,
    # format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    # datefmt="%Y-%m-%d %H:%M:%S",
    # 
    # logger = logging.getLogger(__name__)
    # log_level = os.getenv("PYTHON_LOG_LEVEL", "INFO").upper()
    # logger.setLevel(log_level)
    # # # ─── configure root logger ─────────────────────────────────────────────────────

    import argparse
    
    parser = argparse.ArgumentParser(description="Create data loaders for power network data")
    parser.add_argument("--base_dir", type=str,
                        #default=r"data\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32/test",
                        default = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_val_real__range-30-230_nTest-1000_nVal-1000_2552025_1\test",
                        help="Base directory containing the train/validation folders")
    parser.add_argument("--secondary_dir", type=str,
                         #default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_data_set_test",
                        default=r"data\test_val_real__range-30-150_nTest-10_nVal-10_2732025_32",
                        help="Secondary directory containing the test/validation folders")
    parser.add_argument("--dataset_type", type=str, default="default", 
                        choices=["default", "graphyr", "pinn","cvx"],
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
    
    print("base_dir:", args.base_dir)
    print("secondary_dir:", args.secondary_dir)
    # Create data loaders
    train_loader, val_synthetic_loader,val_real_loader, test_loader = create_data_loaders(
        base_directory=args.base_dir,
        secondary_directory=args.secondary_dir,
        dataset_type=args.dataset_type,
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

        if hasattr(batch, 'batch'):
            # PyG Batch: batch.batch maps each node to its graph-id
            nodes_per_graph = torch.bincount(batch.batch)
            edge_src_to_graph = batch.batch[batch.edge_index[0]]
            edges_per_graph = torch.bincount(edge_src_to_graph)
        else:
            # e.g. if batch is a list of Data objects
            data_list = batch if isinstance(batch, list) else batch.to_data_list()
            nodes_per_graph = torch.tensor([data.num_nodes for data in data_list])
            edges_per_graph = torch.tensor([data.num_edges for data in data_list])

        print(f"Nodes per graph in this batch: {nodes_per_graph.tolist()}")
        print(f"Edges per graph in this batch: {edges_per_graph.tolist()}")
        # Print loader-specific features
        if args.secondary_dir:         
            print("test_batch:", batch_test) 
            print(f"Test Batch size: {len(batch_test)}")
    
    logger.info("Data loaders created successfully.")
