import os
import json
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader, DynamicBatchSampler
import networkx as nx
import numpy as np
import pandapower as pp
from pandapower import from_json, from_json_dict
import logging
import sys
import random
import math
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Configure logger
logger = logging.getLogger(__name__)

CACHE_ROOT_DIR = Path("data/cached_datasets")


def load_pp_networks(base_directory):
    """
    Load pandapower networks from 'mst' and 'mst_opt' subfolders under base_directory.
    """
    nets = {"mst": {}, "mst_opt": {}}
    for phase in ["mst", "mst_opt"]:
        folder = os.path.join(base_directory, phase)
        if not os.path.isdir(folder):
            continue
        for fn in tqdm(os.listdir(folder), desc=f"Loading {phase} networks"):
            if not fn.endswith(".json"): continue
            path = os.path.join(folder, fn)
            try:
                net = pp.from_json(path)
            except Exception:
                with open(path) as f:
                    raw = f.read()
                if raw.startswith('"') and raw.endswith('"'):
                    raw = json.loads(raw)
                try:
                    net = pp.from_json_string(raw)
                except Exception:
                    net = from_json_dict(json.loads(raw))
            if net.bus.empty:
                continue
            nets[phase][fn] = net
    return nets


def create_pyg_from_pp(pp_net_raw):
    """
    Convert a pandapower Net  into a PyG Data object with x, edge_index, edge_attr.
    """
    # Ensure net object
    if isinstance(pp_net_raw, str):
        pp_net = pp.from_json_string(pp_net_raw)
    elif isinstance(pp_net_raw, dict):
        pp_net = from_json_dict(pp_net_raw)
    else:
        pp_net = pp_net_raw

    bus_ids = pp_net.bus.index.to_numpy(dtype=int)
    id2row = {bid: i for i, bid in enumerate(bus_ids)}
    bus_res = pp_net.res_bus.loc[bus_ids]
    x = torch.tensor(np.vstack([
        bus_res.p_mw.values,
        bus_res.q_mvar.values,
        bus_res.vm_pu.values,
        bus_res.va_degree.values
    ]).T, dtype=torch.float)

    # Edge index
    lines = pp_net.line
    from_b = lines.from_bus.map(id2row).to_numpy(dtype=np.int64)
    to_b = lines.to_bus.map(id2row).to_numpy(dtype=np.int64)
    edge_index = torch.from_numpy(np.vstack((from_b, to_b)))

    # Switch state handling
    n_lines = len(lines)
    switch_state = np.ones(n_lines, dtype=int)
    ls = pp_net.switch[pp_net.switch.et == "l"]
    ls_unique = ls.sort_index().drop_duplicates(subset="element", keep="first")
    line_idx = lines.index.to_numpy(dtype=int)
    id2pos = {lid: pos for pos, lid in enumerate(line_idx)}
    elems = ls_unique.element.to_numpy(dtype=int)
    positions = np.array([id2pos[e] for e in elems], dtype=int)
    switch_state[positions] = ls_unique.closed.to_numpy(dtype=int)

    # Edge attributes: R, X, switch
    R = lines.r_ohm_per_km.to_numpy()
    X = lines.x_ohm_per_km.to_numpy()
    edge_attr = torch.from_numpy(np.vstack([R, X, switch_state]).T).float()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))


def process_single_graph(gid, pp_all):
    """
    Build a PyG Data object for graph id 'gid' from loaded pandapower nets.
    """
    net_opt = pp_all["mst_opt"].get(gid)
    if net_opt is None:
        logger.warning(f"No optimized net for {gid}, skipping")
        return None

    data_x = create_pyg_from_pp(pp_all["mst"][gid])
    data_y = create_pyg_from_pp(net_opt)
    data_x.edge_y = data_y.edge_attr[:, 2]  
    data_x.node_y_voltage = data_y.x[:, 2]   

    # Sanity checks
    if data_x.num_nodes != data_y.num_nodes:
        raise ValueError(f"{gid}: node count mismatch X={data_x.num_nodes} vs Y={data_y.num_nodes}")
    if data_x.edge_index.size(1) != data_y.edge_index.size(1):
        raise ValueError(f"{gid}: edge count mismatch X={data_x.edge_index.size(1)} vs Y={data_y.edge_index.size(1)}")

    return data_x


def create_pyg_dataset(base_directory, multiprocessing=True, seed=None):
    pp_all = load_pp_networks(base_directory)
    if seed is not None:
        random.seed(seed)

    graph_ids = list(pp_all["mst"].keys())
    data_list = []

    if multiprocessing:
        num_workers = os.cpu_count() or 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_graph, gid, pp_all): gid for gid in graph_ids}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing graphs"):
                res = future.result()
                if res is not None:
                    data_list.append(res)
    else:
        for gid in tqdm(graph_ids, desc="Processing graphs"):
            res = process_single_graph(gid, pp_all)
            if res is not None:
                data_list.append(res)

    return data_list


def _get_cache_path_suffix(path_str):
    parts = Path(path_str).parts
    if parts and parts[0].lower() == "data":
        return Path(*parts[1:])
    return Path(*parts)


def _load_or_create_dataset(name, in_path, cache_path, multiprocessing, seed):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            data = torch.load(cache_path)
            return list(data)
        except Exception:
            pass
    data = create_pyg_dataset(in_path, multiprocessing, seed)
    torch.save(data, cache_path)
    return data


def create_data_loaders(folder_names, dataset_names, batch_size=32,
                        max_nodes=1000, max_edges=5000,
                        seed=0, multiprocessing=True,
                        num_workers=0, batching_type="standard"):
    torch.manual_seed(seed)
    random.seed(seed)
    CACHE_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {}
    loaders = {}

    for folder, name in zip(folder_names, dataset_names):
        cache_dir = CACHE_ROOT_DIR / _get_cache_path_suffix(folder)
        cache_file = cache_dir / f"{name}.pt"
        ds = _load_or_create_dataset(name, folder, cache_file, multiprocessing, seed)
        datasets[name] = ds

    def _make_loader(ds):
        if batching_type == "dynamic":
            sampler = DynamicBatchSampler(ds, max_nodes, mode="node", shuffle=True,
                                          num_steps=math.ceil(sum(d.num_nodes for d in ds)/max_nodes))
            return DataLoader(ds, batch_sampler=sampler,
                              num_workers=num_workers, pin_memory=False,
                              persistent_workers=(num_workers>0))
        if batching_type == "neighbor":
            return NeighborLoader(ds, num_neighbors=[15,10], batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=False,
                          persistent_workers=(num_workers>0))

    for name, ds in datasets.items():
        loaders[name] = _make_loader(ds)

    return loaders


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    import argparse
    parser = argparse.ArgumentParser("Data loader for power networks")
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--max_edges", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multiprocessing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batching", choices=["standard","dynamic","neighbor"], default="dynamic")
    args = parser.parse_args()

    loaders = create_data_loaders(
        folder_names=args.folders,
        dataset_names=args.names,
        batch_size=args.batch_size,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        seed=args.seed,
        multiprocessing=args.multiprocessing,
        num_workers=args.num_workers,
        batching_type=args.batching
    )
    print("Data loaders created:", list(loaders.keys()))
