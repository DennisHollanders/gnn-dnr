import os
import json
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader, DynamicBatchSampler
import networkx as nx
import numpy as np
import pandapower as pp
from preprocess_data import *
from pandapower import from_json, from_json_dict
import logging 
import sys
import random
import math
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Add necessary source paths
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_search"))
if src_path not in sys.path:
    sys.path.append(src_path)

from preprocess_data import * 

CACHE_ROOT_DIR = Path("data/cached_datasets")
logger = logging.getLogger(__name__)


def load_pp_networks(base_directory):
    nets = {"mst": {}, "mst_opt": {}}
    for phase in ["mst", "mst_opt"]:
        logger.info(f"Loading {phase} networks from {base_directory}")
        folder = os.path.join(base_directory, phase, "pandapower_networks")
        if not os.path.isdir(folder):
            continue
        for fn in tqdm(os.listdir(folder), desc=f"Loading {phase} networks from {folder}"):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(folder, fn)
            try:
                net = pp.from_json(path)
            except:
                with open(path) as f:
                    raw = f.read()
                if raw.startswith('"') and raw.endswith('"'):
                    raw = json.loads(raw)
                try:
                    net = pp.from_json_string(raw)
                except:
                    net = from_json_dict(json.loads(raw))
            if net.bus.empty:
                continue
            nets[phase][fn] = net
    return nets


def create_pyg_from_pp(pp_net_raw):
    """
    Accepts either a pandapower Net, a JSON string, or a dict.
    Converts to a Net if needed, then extracts features and
    a deduplicated line-switch state.
    """
    # --- ensure we have a Net object ---
    if isinstance(pp_net_raw, str):
        pp_net = pp.from_json_string(pp_net_raw)
    elif isinstance(pp_net_raw, dict):
        pp_net = from_json_dict(pp_net_raw)
    else:
        pp_net = pp_net_raw

    # --- bus lookup ---
    bus_ids = pp_net.bus.index.to_numpy(dtype=int)
    id2row  = {bid: i for i, bid in enumerate(bus_ids)}

    # --- node features ---
    bus_res = pp_net.res_bus.loc[bus_ids]
    x = torch.tensor(np.vstack([
        bus_res.p_mw.values,
        bus_res.q_mvar.values,
        bus_res.vm_pu.values,
        bus_res.va_degree.values
    ]).T, dtype=torch.float)

    # --- prepare line edges ---
    lines    = pp_net.line
    n_lines  = len(lines)
    from_b   = lines.from_bus.map(id2row).to_numpy(dtype=np.int64)
    to_b     = lines.to_bus.map(id2row).to_numpy(dtype=np.int64)
    edge_index = torch.from_numpy(np.vstack((from_b, to_b)))

    # --- deduplicate line switches ---
    switch_state = np.ones(n_lines, dtype=int)
    ls = pp_net.switch[pp_net.switch.et == "l"]

    # check for any line with conflicting switch states
    conflicts = (
        ls.groupby("element")["closed"]
          .nunique()
          .loc[lambda s: s > 1]
    )
    if not conflicts.empty:
        logger.warning(
            f"Found conflicting switch states on lines: {conflicts.index.tolist()}"
        )

    # keep only one switch per line (first by switch-ID)
    ls_unique = ls.sort_index().drop_duplicates(subset="element", keep="first")

    # map line-IDs to row positions
    line_idx = lines.index.to_numpy(dtype=int)
    id2pos   = {lid: pos for pos, lid in enumerate(line_idx)}
    elems    = ls_unique.element.to_numpy(dtype=int)
    positions = np.array([id2pos[e] for e in elems], dtype=int)

    # assign closed/open
    switch_state[positions] = ls_unique.closed.to_numpy(dtype=int)

    # --- line features ---
    R = lines.r_ohm_per_km.to_numpy()
    X = lines.x_ohm_per_km.to_numpy()
    edge_attr = torch.from_numpy(np.vstack([R, X, switch_state]).T).float()

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=x.size(0)
    )

def process_single_graph(gid, pp_all, dataset_type):
    logger = logging.getLogger(__name__)
    net_opt = pp_all["mst_opt"].get(gid)
    if net_opt is None:
        logger.warning(f"No optimized net for {gid}, skipping")
        return None


    phase = "mst" 
    data_x = create_pyg_from_pp(pp_all[phase][gid])

    # 2) build y-labels from optimized net
    data_y = create_pyg_from_pp(net_opt)
    data_x.edge_y = data_y.edge_attr[:, 2]    # switch_state
    data_x.node_y_voltage = data_y.x[:, 2]    # vm_pu
    data_x.graph_id = gid
    
    if dataset_type == "cvx":
        try:
            cvx_feat = cxv_features(pp_all[phase][gid])

            N, E = cvx_feat['N'], cvx_feat['E']

            data_x.cvx_N = torch.tensor([N], dtype=torch.long)
            data_x.cvx_E = torch.tensor([E], dtype=torch.long) 
            data_x.cvx_bigM_v = torch.tensor([float(cvx_feat['bigM_v'])], dtype=torch.float32)

            data_x.cvx_from_idx = cvx_feat['from_idx'].unsqueeze(0)
            data_x.cvx_to_idx = cvx_feat['to_idx'].unsqueeze(0)
            data_x.cvx_r_pu = cvx_feat['r_pu'].unsqueeze(0)
            data_x.cvx_x_pu = cvx_feat['x_pu'].unsqueeze(0)
            data_x.cvx_p_inj = cvx_feat['p_inj'].unsqueeze(0)
            data_x.cvx_q_inj = cvx_feat['q_inj'].unsqueeze(0)
            data_x.cvx_bigM_flow = cvx_feat['bigM_flow'].unsqueeze(0)
            data_x.cvx_y0 = cvx_feat['y0'].unsqueeze(0)
            
            sub_idx = cvx_feat['sub_idx']
            data_x.cvx_sub_idx = sub_idx
            # if len(sub_idx) == 0:
            #     data_x.cvx_sub_idx = torch.empty((1, 0), dtype=torch.long)
            # else:
            #     data_x.cvx_sub_idx = sub_idx.unsqueeze(0)
                
        except Exception as e:
            logger.error(f"CVX processing failed for {gid}: {e}")
            return None
    
    return data_x

def create_pyg_dataset(
    base_directory,
    dataset_type: str ="default",
    multiprocessing: bool = True,
    seed: int = None
):
    if dataset_type == "cvx":
        logger.info("including cvx features")

    pp_all = load_pp_networks(base_directory)
    if seed is not None:
        random.seed(seed)

    data_list = []
    graph_ids = list(pp_all["mst"].keys())
    
    if multiprocessing:
        num_workers = min(os.cpu_count(), 1)
        logger.info(f"Using {num_workers} CPU workers for multiprocessing")
        logger.info("Using multiprocessing to create Data objects")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_gid = {
                executor.submit(process_single_graph, gid, pp_all, dataset_type): gid
                for gid in graph_ids
            }
            
            # Process completed futures with tqdm
            for future in tqdm(
                concurrent.futures.as_completed(future_to_gid), 
                total=len(future_to_gid), 
                desc="Processing graphs",
                unit="graph"
            ):
                gid = future_to_gid[future]
                try:
                    data_x = future.result()
                    if data_x is not None:
                        data_list.append(data_x)
                except Exception as e:
                    logger.error(f"Error processing graph {gid}: {e}")
    else: 
        logger.info("Using single-threaded processing to create Data objects")
        for gid, net_orig in tqdm(pp_all["mst"].items(), desc=f"Creating pyg data from {base_directory}"):
            data_x = process_single_graph(gid, pp_all, dataset_type)
            if data_x is not None:
                data_list.append(data_x)

    logger.info(f"Built {len(data_list)} ")
    return data_list

def create_dynamic_loader(dataset, max_nodes=1000, max_edges=5000, shuffle=True, **kwargs):
    est_batches = math.ceil(sum(d.num_nodes for d in dataset) / max_nodes)
    batch_sampler = DynamicBatchSampler(dataset, max_nodes,mode="node", shuffle=shuffle, num_steps =est_batches)
    return DataLoader(
        dataset, 
        batch_sampler=batch_sampler, 
        **kwargs
    )
def create_neighbor_loaders(dataset, num_neighbors=[15, 10], batch_size=1024,shuffle=True, **kwargs):
    return NeighborLoader(
        dataset,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
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
    multiprocessing: bool = True,
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
        created_data = create_pyg_dataset(input_data_path_str, dataset_type, seed=seed, multiprocessing=multiprocessing)
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
    dataset_names: list[str],
    folder_names: list[str],
    dataset_type: str = "default",	
    batch_size: int = 32,
    max_nodes: int = 1000,
    max_edges: int = 5000,
    transform=None,
    train_ratio: float = 0.85,
    seed: int = 0,
    multiprocessing: bool = True,
    num_workers: int = 0,
    batching_type: str = "standard",
    shuffle: bool = True
):
    torch.manual_seed(seed)
    random.seed(seed)

    CACHE_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {dataset_name: None for dataset_name in dataset_names}
    dataloaders = {dataset_name: None for dataset_name in dataset_names}

    def _create_loader(dataset): 
        if not dataset: return None
        loader_kwargs = {
            'num_workers': num_workers,
            'pin_memory': False,  
            'persistent_workers': True if num_workers > 0 else False
        }
        if batching_type == "dynamic":
            return create_dynamic_loader(dataset, max_nodes=max_nodes, max_edges=max_edges, shuffle=shuffle, **loader_kwargs)
        elif batching_type == "standard":
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,**loader_kwargs)
        elif batching_type == "neighbor":
            return create_neighbor_loaders(dataset, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
        return None

    for folder_name, dataset_name in zip(folder_names, dataset_names):
        logger.info(f"Processing folder: {folder_name} for dataset names: {dataset_name}")
        assert os.path.exists(folder_name), f"Folder {folder_name} does not exist."
        assert "mst" in os.listdir(folder_name), f"Folder {folder_name} does not contain 'mst' subfolder."
        assert "mst_opt" in os.listdir(folder_name), f"Folder {folder_name} does not contain 'mst_opt' subfolder."
        
        cache_dir = CACHE_ROOT_DIR / _get_cache_path_suffix(folder_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file_path = cache_dir / f"{dataset_name}-{dataset_type}.pt"
        logger.info(f"Cache file path: {cache_file_path}")
        dataset = _load_or_create_dataset(
            dataset_name=dataset_name,
            input_data_path_str=folder_name,
            dataset_cache_file_path=cache_file_path,
            dataset_type=dataset_type,
            multiprocessing=multiprocessing,  
            seed=seed
        )
        datasets[dataset_name] = dataset
    if dataset_type == "cvx":
        all_data = []
        for ds in datasets.values():
            if ds:
                all_data.extend(ds)
        if not all_data:
            raise ValueError("No data found in any split.")
        # Extract integer values from tensors if needed
        max_N = max(
            (d.cvx_N.item() if isinstance(d.cvx_N, torch.Tensor) else d.cvx_N)
            for d in all_data
        )
        max_E = max(
            (d.cvx_E.item() if isinstance(d.cvx_E, torch.Tensor) else d.cvx_E)
            for d in all_data
        )
        logger.info(f" \n \n \n ======================================\n \n \n")
        logger.info(f"Max nodes (N): {max_N}, Max edges (E): {max_E}")


        # Helper for padding 1D tensors
        def pad_1d_tensor(t: torch.Tensor, L: int, fill: float):
            cur_shape = t.shape
            if t.ndim == 1:
                cur = cur_shape[0]
                if cur == L:
                    return t
                elif cur < L:
                    pad = t.new_full((L - cur,), fill)
                    return torch.cat([t, pad], dim=0)
                else:
                    return t[:L]
            
            elif t.ndim == 2:
                # assume shape (1, Ei) or (batch, Ei)
                batch_dim, cur = cur_shape
                if cur == L:
                    return t
                elif cur < L:
                    # pad along dim=1 to reach (batch_dim, L)
                    pad = t.new_full((batch_dim, L - cur), fill)
                    return torch.cat([t, pad], dim=1)
                else:
                    # slice off extra columns
                    return t[:, :L]
            
            else:
                raise ValueError(f"pad_1d_tensor only supports 1D or 2D, got ndim={t.ndim}")

        max_substations = max(
            len(d.cvx_sub_idx) if hasattr(d, 'cvx_sub_idx') else 0
            for d in all_data
        )
        logger.info(f"Max substations: {max_substations}")
        
        for ds_name, ds in datasets.items():
            if ds is None: continue
            logger.info(f"Padding dataset: {ds_name}")
            for data in tqdm(ds, desc=f"Padding {ds_name}"):
                Ni = data.cvx_N.item()
                Ei = data.cvx_E.item()

                # --- Create node and edge masks ---
                node_mask = torch.zeros(max_N, dtype=torch.bool)
                node_mask[:Ni] = True
                data.cvx_node_mask = node_mask.unsqueeze(0)

                edge_mask = torch.zeros(max_E, dtype=torch.bool)
                edge_mask[:Ei] = True
                data.cvx_edge_mask = edge_mask.unsqueeze(0)

                # --- Create substation and non-substation masks ---
                sub_mask = torch.zeros(max_N, dtype=torch.float32)
                sub_indices = data.cvx_sub_idx.long()  # No squeeze needed now
                valid_sub_indices = sub_indices[sub_indices < Ni]
                if len(valid_sub_indices) > 0:
                    sub_mask[valid_sub_indices] = 1.0
                data.cvx_sub_mask = sub_mask.unsqueeze(0)

                non_sub_mask = node_mask.float() - sub_mask
                data.cvx_non_sub_mask = non_sub_mask.unsqueeze(0)

                # --- Pad existing features and create pre-calculated ones ---
                data.cvx_r_pu      = pad_1d_tensor(data.cvx_r_pu, max_E, 0.0)
                data.cvx_x_pu      = pad_1d_tensor(data.cvx_x_pu, max_E, 0.0)
                data.cvx_p_inj     = pad_1d_tensor(data.cvx_p_inj, max_N, 0.0)
                data.cvx_q_inj     = pad_1d_tensor(data.cvx_q_inj, max_N, 0.0)
                data.cvx_y0        = pad_1d_tensor(data.cvx_y0, max_E, 0.0)
                data.cvx_bigM_flow = pad_1d_tensor(data.cvx_bigM_flow, max_E, 0.0)
                from_idx_padded    = pad_1d_tensor(data.cvx_from_idx.long(), max_E, 0)
                to_idx_padded      = pad_1d_tensor(data.cvx_to_idx.long(), max_E, 0)

                # --- Create and add pre-calculated squared parameters ---
                data.cvx_bigM_flow_sq = pad_1d_tensor(data.cvx_bigM_flow.pow(2), max_E, 0.0)
                z_line_sq = data.cvx_r_pu.pow(2) + data.cvx_x_pu.pow(2)
                data.cvx_z_line_sq = pad_1d_tensor(z_line_sq, max_E, 0.0)

                # --- Create incidence matrix parameters ---
                A_from = torch.zeros((max_E, max_N), dtype=torch.float32)
                A_to   = torch.zeros((max_E, max_N), dtype=torch.float32)

                active_edge_indices = torch.arange(Ei)
                A_from[active_edge_indices, from_idx_padded.squeeze(0)[:Ei]] = 1.0
                A_to[active_edge_indices, to_idx_padded.squeeze(0)[:Ei]] = 1.0
                data.cvx_A_from = A_from.unsqueeze(0)
                data.cvx_A_to = A_to.unsqueeze(0)

                # --- Pad sub_idx to consistent size ---
                if max_substations > 0:
                    num_subs = len(sub_indices)
                    if num_subs < max_substations:
                        # Pad with -1 or max_N to indicate invalid indices
                        padding = torch.full((max_substations - num_subs,), -1, dtype=torch.long)
                        sub_indices_padded = torch.cat([sub_indices, padding])
                    else:
                        sub_indices_padded = sub_indices[:max_substations]
                else:
                    sub_indices_padded = torch.empty(0, dtype=torch.long)
                
                data.cvx_sub_idx = sub_indices_padded.unsqueeze(0)

                # --- Finally, overwrite old size tensors ---
                data.cvx_N = torch.tensor([max_N], dtype=torch.long)
                data.cvx_E = torch.tensor([max_E], dtype=torch.long)
                data.cvx_from_idx = from_idx_padded
                data.cvx_to_idx = to_idx_padded
    for dataset_name, dataset in datasets.items():
        if dataset:
            dataloaders[dataset_name] = _create_loader(dataset)
    print(f"\nCreated data loaders with:")
    print(f"  Dataset names: {dataset_names}")
    print(f"  Folder names: {folder_names}")
    print(f"  Dataset type: {dataset_type}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max nodes: {max_nodes}")
    print(f"  Max edges: {max_edges}")

    return dataloaders


if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",)
    
    logger = logging.getLogger(__name__)
    log_level = os.getenv("PYTHON_LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)
    # # # ─── configure root logger ─────────────────────────────────────────────────────

    import argparse
    
    parser = argparse.ArgumentParser(description="Create data loaders for power network data")
    parser.add_argument("--dataset_names", type=str, nargs="+", default= [
                                                                        "train",
                                                                         "validation",
                                                                         "test",
                                                                          ]
                                                                          , help="Names of datasets to create loaders for")
    parser.add_argument("--folder_names", type=str, nargs="+", default=[
                r"data\split_datasets\train",
                r"data\split_datasets\validation",
                r"data\split_datasets\test",]
                , help="Names of folders to look for datasets in")
    parser.add_argument("--dataset_type", type=str, default="cvx", 
                        choices=["default", "cvx"],
                        help="Type of dataloader to create")
    parser.add_argument("--batching_type", type=str, default="standard",
                        choices =["standard", "dynamic", "neighbor"],)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_nodes", type=int, default=10000, help="Maximum number of nodes in a batch (for dynamic batching)")
    parser.add_argument("--max_edges", type=int, default=50000, help="Maximum number of edges in a batch (for dynamic batching)")
    parser.add_argument("--train_ratio", type=float, default=0.85, help="Ratio of training set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting")
    parser.add_argument("--multiprocessing", default=False, help="Use multiprocessing for dataset creation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    logger.info(f"Creating data loaders with the following parameters:")
    logger.info(f"  Dataset names: {args.dataset_names}")
    logger.info(f"  Folder names: {args.folder_names}")
    # Create data loaders
    data_loaders = create_data_loaders(
        dataset_names=args.dataset_names,
        folder_names=args.folder_names,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        train_ratio=args.train_ratio,
        seed=args.seed,
        multiprocessing =args.multiprocessing,
        num_workers=args.num_workers,
        batching_type = args.batching_type,
    )
    train_loader = data_loaders.get("train")
    test_loader = data_loaders.get("test")
    validation_loader = data_loaders.get("validation")
    try:
        validation_real_loader = data_loaders.get("validation_real")
    except:
        pass
    
    print("\nData loaders created successfully.")

    # Print sample batch information
    if train_loader:
        print("\nSample batch information:")
        if isinstance(train_loader, list):
            # For NeighborLoader
            batch = next(iter(train_loader[0]))
        else:
            # For regular DataLoader
            batch = next(iter(train_loader))
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


    logger.info("Data loaders created successfully.")
