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
#from MILP_class_dnr import MILP_class  
from load_data import load_graph_data


def validate_and_store_optimized_model(optimizer, net, graph_id, logger,
                                         storage_folder=Path("optimized_models"),
                                         voltage_lower=0.9, voltage_upper=1.10):
    """
    Validates optimization outcomes and stores the updated pandapower network and a networkx graph.
    """
    # Ensure the storage folder exists.
    storage_folder.mkdir(parents=True, exist_ok=True)

    # Extract optimization results.
    opt_results = optimizer.extract_results()
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

    # Update the pandapower network with optimized switch statuses.
    net_updated = optimizer.update_network()

    # Save the updated pandapower network as JSON.
    try:
        pp_file = storage_folder / f"{graph_id}_optimized.json"
        net_json = pp.to_json(net_updated)
        with open(pp_file, "w") as f:
            f.write(net_json)
        logger.info(f"Graph {graph_id}: Optimized pandapower network stored in {pp_file}")
    except Exception as e:
        logger.error(f"Graph {graph_id}: Error saving pandapower network - {e}")

    # Create a networkx graph representation from the updated network.
    try:
        G = nx.Graph()
        # Add all buses as nodes.
        for bus in net_updated.bus.index:
            G.add_node(bus)
        # Add edges for each line that is in service (all related switches closed).
        for idx, line in net_updated.line.iterrows():
            add_edge = True
            if hasattr(net_updated, "switch") and not net_updated.switch.empty:
                switches = net_updated.switch[(net_updated.switch.et == 'l') & (net_updated.switch.element == idx)]
                if not switches.empty and not all(switches["closed"]):
                    add_edge = False
            if add_edge:
                G.add_edge(line["from_bus"], line["to_bus"])
        nx_file = storage_folder / f"{graph_id}_optimized.graphml"
        nx.write_graphml(G, str(nx_file))
        logger.info(f"Graph {graph_id}: Optimized networkx graph stored in {nx_file}")
    except Exception as e:
        logger.error(f"Graph {graph_id}: Error creating networkx graph - {e}")
        G = None

    return net_updated, G


def apply_optimization_and_store_ground_truths(folder_path, method="SOCP", toggles=None, debug=False, logger=None):
    folder_path = Path(folder_path)

    # Directories for ground truth data
    pp_gt_dir = folder_path / "pandapower_gt"
    feat_gt_dir = folder_path / "features_gt"
    pp_gt_dir.mkdir(parents=True, exist_ok=True)
    feat_gt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nApplying {method.upper()} optimization on networks in {folder_path}...\n")

    # Load data
    _, pp_networks, features = load_graph_data(folder_path)

    metrics_data = []

    if isinstance(pp_networks, str):
        pp_networks = pp.from_json_string(pp_networks)
        print("Loaded network data from JSON string")   

    for graph_id, net in pp_networks.items():
        try: 
            print(f"\n{'='*30} Processing {graph_id} {'='*30}")
    
            if isinstance(net, str):
                net = pp.from_json_string(net)
                print("In loop: loaded network as string.")

            # Run a power flow to check convergence.
            pp.runpp(net, enforce_q_lims=False)
            if not net.converged:
                print(f"Power flow did not converge for {graph_id}. Skipping optimization.")
                continue

            original_switch_states = net.switch["closed"].copy() if hasattr(net, "switch") else None
            original_vm_pu = net.res_bus.vm_pu.copy()

            # Instantiate optimizer based on chosen method.
            #if method.upper() == "MILP":
            #    optimizer = MILP_class(net, graph_id, logger=logger, toggles=toggles)
            #else:
            optimizer = SOCP_class(net, graph_id, logger=logger, toggles=toggles)

            if hasattr(optimizer, 'initialize'):
                optimizer.initialize()
            
            # Call initialize_with_alternative_mst which may alter switch states.
            try:
                optimizer.initialize_with_alternative_mst(penalty=1.0)
            except Exception as e:
                print(f"Error during initialize_with_alternative_mst for {graph_id}: {e}")
            
            # Build and solve the optimization model.
            model = optimizer.create_model()  
            start_time = time.time()
            solver_results = optimizer.solve()
            optimization_time = time.time() - start_time

            num_switches_changed = (
                optimizer.num_switches_changed if hasattr(optimizer, 'num_switches_changed') else 0
            )
            
            # Extract the optimization results.
            opt_results = optimizer.extract_results()
            try:
                print(f"Optimization solved in {optimization_time:.4f} seconds, Switches changed: {num_switches_changed}")
            except AttributeError:
                print("Optimization results not available.")

            # Prepare ground truth features with optimization metadata and predicted voltages.
            features_gt = features.get(graph_id, {}).copy()
            features_gt.update({
                "optimization_time": optimization_time,
                "switches_changed": num_switches_changed,
                "predicted_voltages": opt_results.get("voltage_profiles", {})
            })

            # Save ground truth features.
            with open(feat_gt_dir / f"{graph_id}.pkl", "wb") as f:
                pkl.dump(features_gt, f)
            print(f"Saved ground truth features for {graph_id}")

            # Validate optimization results and store updated models.
            net_updated, nx_graph = validate_and_store_optimized_model(optimizer, net, graph_id, logger)
            
            # Optionally, store the updated pandapower network (e.g., in a ground truth folder).
            updated_pp_file = pp_gt_dir / f"{graph_id}_optimized.json"
            try:
                with open(updated_pp_file, "w") as f:
                    f.write(pp.to_json(net_updated))
                print(f"Saved updated pandapower network for {graph_id}")
            except Exception as e:
                print(f"Error saving updated network for {graph_id}: {e}")

            # Append metrics data for summary (graph_id, optimization_time, number of switches changed)
            metrics_data.append({
                "graph_id": graph_id,
                "optimization_time": optimization_time,
                "switches_changed": num_switches_changed
            })
            
        except: 
            print(f"Error processing {graph_id}: {sys.exc_info()[1]}")
            continue

    # Save optimization metrics.
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv = folder_path / "optimization_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n✓ Saved optimization metrics to {metrics_csv}")

    # Summary printing.
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Total graphs processed: {len(pp_networks)}")
    print(f"Successful optimizations: {len(metrics_data)}")
    if not metrics_df.empty:
        print(f"Average optimization time: {metrics_df['optimization_time'].mean():.4f} seconds")
        print(f"Total switches changed: {metrics_df['switches_changed'].sum()}")
    print("="*50)

    # Plot the distribution of switches changed and optimization time.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Bar plot for number of switches changed per graph.
    axes[0].bar(metrics_df['graph_id'], metrics_df['switches_changed'], color='blue')
    axes[0].set_title('Switches Changed per Graph')
    axes[0].set_xlabel('Graph ID')
    axes[0].set_ylabel('Switches Changed')
    axes[0].tick_params(axis='x', rotation=45)

    # Bar plot for optimization time per graph.
    axes[1].bar(metrics_df['graph_id'], metrics_df['optimization_time'], color='green')
    axes[1].set_title('Optimization Time per Graph (s)')
    axes[1].set_xlabel('Graph ID')
    axes[1].set_ylabel('Time (s)')
    axes[1].tick_params(axis='x', rotation=45)

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
    logging.basicConfig(level=logging.DEBUG)
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_filename = log_dir / f"{method.upper()}_logs.txt"
    logger = logging.getLogger("network_optimizer")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_filename, mode="w")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logger.info(f"Logging initialized. All logs will be stored in {log_filename}")
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ground truth data for power networks using optimization')
    parser.add_argument('--folder_path',
                        default = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\transformed_subgraphs_26032025_4",
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
    
    MILP_toggles = { }
    
    toggles = SOCP_toggles if args.method.upper() == "SOCP" else MILP_toggles

    print("Toggles for optimization:")
    print(toggles)
    if args.set:
        apply_optimization_and_store_ground_truths(Path(args.folder_path) / args.set, method=args.method, toggles=toggles, debug=args.debug, logger=logger)
    elif args.set == "all": 
        for set_name in Path(args.folder_path).iterdir():
            if set_name.is_dir():
                print("\nProcessing set:", set_name)
                apply_optimization_and_store_ground_truths(Path(args.folder_path) / set_name, method=args.method, toggles=toggles, debug=args.debug, logger=logger)
    else:
        apply_optimization_and_store_ground_truths(args.folder_path, method=args.method, toggles=toggles, debug=args.debug, logger=logger)

    print("\nGround truth generation complete!!!!")
