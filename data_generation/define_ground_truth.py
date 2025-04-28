import json
import pandas as pd
import pandapower as pp
import pickle as pkl
import numpy as np
import os
import sys 
import time
from pathlib import Path
import argparse
import networkx as nx
import logging
import matplotlib.pyplot as plt

# Add necessary source paths
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
load_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_search"))
sys.path.extend([src_path, load_data_path])

from SOCP_class_dnr import SOCP_class
from load_data import load_graph_data

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def get_n_workers():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

def validate_and_store_optimized_model(optimizer, net, graph_id, logger,
                                         root_folder=Path("dataset_folder"), 
                                         voltage_lower=0.9, voltage_upper=1.10):
    """
    Validates optimization outcomes and stores the updated pandapower network and a networkx graph
    in the folder structure: root_folder/Original, Post_MST, Optimization.
    """
    # Define the folder structure.
    original_folder = root_folder / "original"
    post_mst_folder = root_folder / "post_MST"
    optimization_folder = root_folder / "optimization"

    # Create necessary directories.
    original_folder.mkdir(parents=True, exist_ok=True)
    post_mst_folder.mkdir(parents=True, exist_ok=True)
    optimization_folder.mkdir(parents=True, exist_ok=True)

    # Subfolders for features, networkx, and pandapower graphs
    for folder in [original_folder, post_mst_folder, optimization_folder]:
        (folder / "graph_features").mkdir(parents=True, exist_ok=True)
        (folder / "networkx_graphs").mkdir(parents=True, exist_ok=True)
        (folder / "pandapower_networks").mkdir(parents=True, exist_ok=True)

    voltage_profiles = opt_results.get("voltage_profiles", {})
    switches_changed = opt_results.get("num_switches_changed", None)
    obj_value = opt_results.get("objective_value", None)

    # Validate voltage profiles.
    for bus, voltage in voltage_profiles.items():
        if voltage < voltage_lower or voltage > voltage_upper:
            logger.warning(f"Graph {graph_id}: Bus {bus}: voltage {voltage:.4f} p.u. out of bounds [{voltage_lower}, {voltage_upper}].")

    # Log switch change information.
    if switches_changed is not None:
        logger.info(f"Graph {graph_id}: {switches_changed} switches changed.")
    else:
        logger.warning(f"Graph {graph_id}: Switch change data is unavailable.")

    # Check for potential infeasibility (e.g., extremely high objective value).
    if optimizer.optimized_results is None or (obj_value is not None and obj_value > 1e6):
        logger.error(f"Graph {graph_id}: Optimization may be infeasible (objective value: {obj_value}).")


    # Save Original pandapower network.
    try:
        original_pp_dir = original_folder / "pandapower_networks"
        pp_file = original_pp_dir / f"{graph_id}_original.json"
        net_json = pp.to_json(net)
        with open(pp_file, "w") as f:
            f.write(net_json)
        logger.info(f"Graph {graph_id}: Original pandapower network stored in {pp_file}")
    except Exception as e:
        logger.error(f"Graph {graph_id}: Error saving original pandapower network - {e}")

    # Save Original networkx graph.
    try:
        original_nx_dir = original_folder / "networkx_graphs"
        G = nx.Graph()
        for bus in net.bus.index:
            G.add_node(bus)
        for idx, line in net.line.iterrows():
            add_edge = True
            if hasattr(net, "switch") and not net.switch.empty:
                switches = net.switch[(net.switch.et == 'l') & (net.switch.element == idx)]
                if not switches.empty and not all(switches["closed"]):
                    add_edge = False
            if add_edge:
                G.add_edge(line["from_bus"], line["to_bus"])
        nx_file = original_nx_dir / f"{graph_id}_original.graphml"
        nx.write_graphml(G, str(nx_file))
        logger.info(f"Graph {graph_id}: Original networkx graph stored in {nx_file}")
    except Exception as e:
        logger.error(f"Graph {graph_id}: Error creating networkx graph - {e}")

    # Save features for original graph.
    try:
        original_feat_dir = original_folder / "graph_features"
        features_file = original_feat_dir / f"{graph_id}_features.pkl"
        with open(features_file, "wb") as f:
            pkl.dump(opt_results, f)
        logger.info(f"Graph {graph_id}: Features stored in {features_file}")
    except Exception as e:
        logger.error(f"Graph {graph_id}: Error saving features - {e}")


    # Save Post-MST pandapower network.
    try:
        net.switch["closed"] = optimizer.switch_df["closed"]
        pp.runpp(net, enforce_q_lims=False)
        post_mst_pp_dir = post_mst_folder / "pandapower_networks"
        pp_file = post_mst_pp_dir / f"{graph_id}_post_mst.json"
        net_json = pp.to_json(net)
        with open(pp_file, "w") as f:
            f.write(net_json)
        logger.info(f"Graph {graph_id}: Post-MST pandapower network stored in {pp_file}")
    except Exception as e:
        logger.error(f"Graph {graph_id}: Error saving post-MST pandapower network - {e}")

    # Save Post-MST networkx graph.
    try:
        post_mst_nx_dir = post_mst_folder / "networkx_graphs"
        G = nx.Graph()
        for bus in net.bus.index:
            G.add_node(bus)
        for idx, line in net.line.iterrows():
            add_edge = True
            if hasattr(net, "switch") and not net.switch.empty:
                switches = net.switch[(net.switch.et == 'l') & (net.switch.element == idx)]
                if not switches.empty and not all(switches["closed"]):
                    add_edge = False
            if add_edge:
                G.add_edge(line["from_bus"], line["to_bus"])
        nx_file = post_mst_nx_dir / f"{graph_id}_post_mst.graphml"
        nx.write_graphml(G, str(nx_file))
        logger.info(f"Graph {graph_id}: Post-MST networkx graph stored in {nx_file}")
    except Exception as e:
        logger.error(f"Graph {graph_id}: Error creating post-MST networkx graph - {e}")

    # Save features for Post-MST.
    try:
        post_mst_feat_dir = post_mst_folder / "graph_features"
        features_file = post_mst_feat_dir / f"{graph_id}_features.pkl"
        with open(features_file, "wb") as f:
            pkl.dump(opt_results, f)
        logger.info(f"Graph {graph_id}: Features stored in {features_file}")
    except Exception as e:
        logger.error(f"Graph {graph_id}: Error saving features - {e}")

    # Save Optimized pandapower network.
    try:
        optimization_pp_dir = optimization_folder / "pandapower_networks"
        pp_file = optimization_pp_dir / f"{graph_id}_optimized.json"
        net_json = pp.to_json(net_updated)
        with open(pp_file, "w") as f:
            f.write(net_json)
        logger.info(f"Graph {graph_id}: Optimized pandapower network stored in {pp_file}")
    except Exception as e:
        logger.error(f"Graph {graph_id}: Error saving optimized pandapower network - {e}")

    # Save Optimized networkx graph.
    try:
        optimization_nx_dir = optimization_folder / "networkx_graphs"
        G = nx.Graph()
        for bus in net_updated.bus.index:
            G.add_node(bus)
        for idx, line in net_updated.line.iterrows():
            add_edge = True
            if hasattr(net_updated, "switch") and not net_updated.switch.empty:
                switches = net_updated.switch[(net_updated.switch.et == 'l') & (net_updated.switch.element == idx)]
                if not switches.empty and not all(switches["closed"]):
                    add_edge = False
            if add_edge:
                G.add_edge(line["from_bus"], line["to_bus"])
        nx_file = optimization_nx_dir / f"{graph_id}_optimized.graphml"
        nx.write_graphml(G, str(nx_file))
        logger.info(f"Graph {graph_id}: Optimized networkx graph stored in {nx_file}")
    except Exception as e:
        logger.error(f"Graph {graph_id}: Error creating optimized networkx graph - {e}")

    # Save features for Optimized graph.
    try:
        optimization_feat_dir = optimization_folder / "graph_features"
        features_file = optimization_feat_dir / f"{graph_id}_features.pkl"
        with open(features_file, "wb") as f:
            pkl.dump(opt_results, f)
        logger.info(f"Graph {graph_id}: Features stored in {features_file}")
    except Exception as e:
        logger.error(f"Graph {graph_id}: Error saving features - {e}")

    return net_updated, G

def process_single_graph(graph_id, net, features, folder_path, method, toggles, logger):
    """
    Run the SOCP/MILP optimization on one network, save ground truths,
    and return a metrics dict or None on failure.
    """
   
    try:
        logger.debug(f"{graph_id}: received net of type {type(net)}")
        if isinstance(net, str):
            logger.debug(f"{graph_id}: deserializing JSON → pandapower network")
            net = pp.from_json_string(net)

        pp.runpp(net, enforce_q_lims=False)
        if not net.converged:
            logger.warning(f"{graph_id}: base PF failed, skipping")
            return None

        original_vm    = net.res_bus.vm_pu.copy()
        original_loss  = net.res_line.pl_mw.sum()
        original_sw    = net.switch["closed"].copy() if hasattr(net, "switch") else None

        # 2) instantiate optimizer
        optimizer = SOCP_class(net, graph_id, logger=logger, toggles=toggles)
        optimizer.initialize()  
        optimizer.initialize_with_alternative_mst(penalty=1.0)
        net.switch["closed"] = optimizer.switch_df["closed"]
        pp.runpp(net, enforce_q_lims=False)
        alt_vm   = net.res_bus.vm_pu.copy()
        alt_loss = net.res_line.pl_mw.sum()

        model = optimizer.create_model()


        # 3) solve
        start = time.time()
        solver_res = optimizer.solve()
        opt_time = time.time() - start
        num_sw   = getattr(optimizer, "num_switches_changed", 0)
        opt_res  = optimizer.extract_results()

        # 4) apply & final PF
        optimizer.update_network()
        pp.runpp(net, enforce_q_lims=False)
        opt_vm   = net.res_bus.vm_pu.copy()
        opt_loss = net.res_line.pl_mw.sum()

        # 5) dump features & networks (you can call your existing validator)
        features_gt = features.get(graph_id, {}).copy()
        features_gt.update({
            "optimization_time": opt_time,
            "switches_changed":  num_sw,
            "original_vm":       original_vm,
            "original_loss":     original_loss,
            "alternative_vm":    alt_vm,
            "alternative_loss":  alt_loss,
            "optimized_vm":      opt_vm,
            "optimized_loss":    opt_loss,
        })
        # save feature pickle
        feat_gt_dir = folder_path / "features_gt"
        with open(feat_gt_dir / f"{graph_id}.pkl", "wb") as f:
            pkl.dump(features_gt, f)
        # store networks & graphs
        validate_and_store_optimized_model(optimizer, net, graph_id, logger,
                                           root_folder=folder_path)

        return {"graph_id": graph_id,
                "optimization_time": opt_time,
                "switches_changed":  num_sw}

    except Exception as e:
        logger.error(f"{graph_id}: processing error: {e}")
        return None
    
def apply_optimization(folder_path, method="SOCP", toggles=None, debug=False, logger=None):
    folder_path = Path(folder_path)

    # Directories for ground truth data
    pp_gt_dir = folder_path / "pandapower_gt"
    feat_gt_dir = folder_path / "features_gt"
    pp_gt_dir.mkdir(parents=True, exist_ok=True)
    feat_gt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nApplying {method.upper()} optimization on networks in {folder_path}...\n")

    # Load data
    _, pp_networks, features = load_graph_data(folder_path)

    if isinstance(pp_networks, str):
        pp_networks = pp.from_json_string(pp_networks)
        print("Loaded network data from JSON string")  

    items = list(pp_networks.items())  # [(graph_id, net), ...]

    n_workers = get_n_workers()
    metrics = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_single_graph,
                graph_id, net, features, folder_path, method, toggles, logger
            ): graph_id
            for graph_id, net in items
        }

        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Generating ground truth"):
            res = fut.result()
            if res is not None:
                metrics.append(res)

    # Save optimization metrics.
    metrics_df = pd.DataFrame(metrics)
    metrics_csv = folder_path / "optimization_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n✓ Saved optimization metrics to {metrics_csv}")

    # Summary printing.
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Total graphs processed: {len(pp_networks)}")
    print(f"Successful optimizations: {len(metrics)}")
    if not metrics_df.empty:
        print(f"Average optimization time: {metrics_df['optimization_time'].mean():.4f} seconds")
        print(f"Total switches changed: {metrics_df['switches_changed'].sum()}")
    print("="*50)


    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    if 'switches_changed' in metrics_df:
        axes[0].hist(metrics_df['switches_changed'], bins=10, alpha=0.7)
        axes[0].set_title('Switches Changed per Graph')
        axes[0].set_xlabel('Switches Changed')
        axes[0].set_ylabel('Frequency')
    else:
        axes[0].text(0.5, 0.5, 'No switch data', ha='center')
        axes[0].set_title('Switches Changed per Graph')

    if 'optimization_time' in metrics_df:
        axes[1].hist(metrics_df['optimization_time'], bins=10, alpha=0.7)
        axes[1].set_title('Optimization Time per Graph (s)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency')
    else:
        axes[1].text(0.5, 0.5, 'No time data', ha='center')
        axes[1].set_title('Optimization Time per Graph')

    plt.tight_layout()
    plt.show()
   

def is_radial_and_connected(net):
    """
    Checks if the pandapower network `net` is radial (i.e. forms a tree)
    and connected (i.e. one connected component).
    """
    G = nx.Graph()
    for bus in net.bus.index:
        G.add_node(bus)
    for idx, line in net.line.iterrows():
        add_edge = True
        if hasattr(net, "switch") and not net.switch.empty:
            switches = net.switch[(net.switch.et == 'l') & (net.switch.element == idx)]
            if not switches.empty and not all(switches["closed"]):
                add_edge = False
        if add_edge:
            G.add_edge(line["from_bus"], line["to_bus"])
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    is_radial = num_edges == num_nodes - 1
    is_connected = nx.is_connected(G)
    return is_radial, is_connected


from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.environ import value, Constraint, Var

def debug_infeasibility(model, tol=1e-6):
    print("=== Infeasible Constraints ===")
    log_infeasible_constraints(model, log_expression=True, tol=tol)

def print_constraint_violations(model, tol=1e-6):
    print("\n=== Constraint Violations ===")
    for constr in model.component_data_objects(Constraint, active=True):
        try:
            lower = constr.lower if constr.lower is not None else -float('inf')
            upper = constr.upper if constr.upper is not None else float('inf')
            body_val = value(constr.body)
            violation = max(lower - body_val, body_val - upper, 0)
            if violation > tol:
                print(f"{constr.name} (index: {constr.index() if hasattr(constr, 'index') else ''}) violation: {violation:.4e}, body value: {body_val:.4e}, bounds: ({lower}, {upper})")
        except Exception as e:
            print(f"Could not evaluate constraint {constr.name}: {e}")

def init_logging(method):
    # 1) Grab your specific logger and configure it only—no basicConfig
    logger = logging.getLogger("network_optimizer")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False              # don’t bubble up to root

    # 2) File handler at DEBUG
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{method.upper()}_logs.txt", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # 3) Stream handler at DEBUG (or INFO if you prefer less console spam)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

    # 4) Silence all matplotlib logging below WARNING
    ml = logging.getLogger("matplotlib")
    ml.setLevel(logging.WARNING)
    mf = logging.getLogger("matplotlib.font_manager")
    mf.setLevel(logging.WARNING)

    logger.debug(f"Logging initialized. All logs to {log_dir}/{method.upper()}_logs.txt")
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ground truth data for power networks using optimization')
    parser.add_argument('--folder_path',
                        default = r"data\transformed_subgraphs_26032025_4",
                        type=str, help='Dataset folder path')
    parser.add_argument('--set', type=str, choices=['test', 'validation', 'train', '', 'all'], default='', help='Dataset set to process; leave empty for no subfolder')
    parser.add_argument('--method', type=str, choices=['SOCP', 'MILP'], default='SOCP', help='Choose optimization method: SOCP or MILP')
    parser.add_argument('--debug', type=bool, default=True, help='Print debug information')

    # SOCP toggles
    parser.add_argument('--include_voltage_drop_constraint', type=bool, default=True, help="Include voltage drop constraint SOCP")
    parser.add_argument('--include_voltage_bounds_constraint', type=bool, default=True, help="Include voltage bounds constraint SOCP")
    parser.add_argument('--include_power_balance_constraint', type=bool, default=True, help="Include power balance constraint SOCP")
    parser.add_argument('--include_radiality_constraints', type=bool, default=True, help="Include radiality constraints SOCP")
    parser.add_argument('--use_spanning_tree_radiality', type=bool, default=False, help="Use spanning tree radiality SOCP")
    parser.add_argument('--include_switch_penalty', type=bool, default=False, help="Include switch penalty in objective SOCP")
    
    parser.add_argument("--write_files", action="store_true", help="If set, write out LP/MPS model files; otherwise skip for speed")
    args = parser.parse_args()

    logger = init_logging(args.method)
    
    SOCP_toggles = { 
        "include_voltage_drop_constraint": args.include_voltage_drop_constraint, 
        "include_voltage_bounds_constraint": args.include_voltage_bounds_constraint,   
        "include_power_balance_constraint": args.include_power_balance_constraint,  
        "include_radiality_constraints": args.include_radiality_constraints,
        "use_spanning_tree_radiality": args.use_spanning_tree_radiality,  
        "include_switch_penalty": args.include_switch_penalty,
    }
    
    print("Toggles for optimization:")
    print(SOCP_toggles)
    if args.set:
        apply_optimization(Path(args.folder_path) / args.set, method=args.method, toggles=SOCP_toggles, debug=args.debug, logger=logger)
    elif args.set == "all": 
        for set_name in Path(args.folder_path).iterdir():
            if set_name.is_dir():
                print("\nProcessing set:", set_name)
                apply_optimization(Path(args.folder_path) / set_name, method=args.method, toggles=SOCP_toggles, debug=args.debug, logger=logger)
    else:
        apply_optimization(args.folder_path, method=args.method, toggles=SOCP_toggles, debug=args.debug, logger=logger)

    print("\nGround truth generation complete!!!!")
