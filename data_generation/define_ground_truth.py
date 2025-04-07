# import json
# import pandas as pd
# import pandapower as pp
# import pickle as pkl
# import numpy as np
# import os
# import sys 
# import time
# from pathlib import Path
# import argparse
# import networkx as nx
# import logging

# # Add necessary source paths
# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
# load_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_search"))
# sys.path.extend([src_path, load_data_path])

# from SOCP_class_dnr import SOCP_class
# from MILP_class_dnr import MILP_class  
# from load_data import load_graph_data

# def apply_optimization_and_store_ground_truths(folder_path, method="SOCP", toggles=None, debug =False, logger=None):
#     folder_path = Path(folder_path)

#     # Directories for ground truth data
#     pp_gt_dir = folder_path / "pandapower_gt"
#     feat_gt_dir = folder_path / "features_gt"
#     pp_gt_dir.mkdir(parents=True, exist_ok=True)
#     feat_gt_dir.mkdir(parents=True, exist_ok=True)

#     print(f"\nApplying {method.upper()} optimization on networks in {folder_path}...\n")

#     # Load data
#     _, pp_networks, features = load_graph_data(folder_path)

#     metrics_data = []

#     if isinstance(pp_networks, str):
#         pp_networks = pp.from_json_string(pp_networks)
#         print("Loaded network data from JSON string")   

#     for graph_id, net in pp_networks.items():
#         print(f"\n{'='*30} Processing {graph_id} {'='*30}")
    
#         if isinstance(net, str):
#             net = pp.from_json_string(net)
#             print("in loop loaded network as string")

#         original_switch_states = net.switch["closed"].copy() if hasattr(net, "switch") else None

#         # check if the network runs correctly! 
#         pp.runpp(net, enforce_q_lims=False)
#         if not net.converged:
#             print(f"Power flow did not converge for {graph_id}. Skipping optimization.")
#             continue

#         # Instantiate optimizer based on chosen method
#         if method.upper() == "MILP":
#             optimizer = MILP_class(net, graph_id, logger= logger, toggles=toggles)
#         else:
#             optimizer = SOCP_class(net, graph_id, logger=logger, toggles=toggles)

#         if hasattr(optimizer, 'initialize'):
#             optimizer.initialize()  
#         try:
#             optimizer.initialize_with_alternative_mst(penalty=1.0)
#         except Exception as e:
#             print(f"Error during initialization with alternative MST for {graph_id}: {e}")
    
#         model = optimizer.create_model() #toggles = toggles)

#         #if debug:
#         #    debug_infeasibility(model)
#         #    print_constraint_violations(model)

        
#         start_time = time.time()
#         results = optimizer.solve()
#         #model.feasRelax()
#         optimization_time = time.time() - start_time

#         #updated_net =optimizer.extract_results()
#         #optimizer.print(results)
#         #is_radial, is_connected = is_radial_and_connected(updated_net)  
#         #print(f"Is radial: {is_radial}, Is connected: {is_connected}")
#         # Calculate switch state changes
#         # num_switches_switched = 0
#         # if original_switch_states is not None:
#         #     num_switches_switched = int((original_switch_states != updated_net.switch["closed"]).sum())
#         # print(f"Optimization solved in {optimization_time:.4f} seconds, Switches changed: {num_switches_switched}")

#         metrics_data.append({
#             "graph_id": graph_id,
#             "optimization_time": optimization_time,
#             #"switches_changed": num_switches_switched
#         })

#         # Update features with ground truths
#         features_gt = features.get(graph_id, {}).copy()
#         node_features = features_gt.get("node_features", {})
#         # for node in updated_net.bus.index:
#         #     if node in node_features:
#         #         node_features[node]["v_gt"] = updated_net.res_bus.vm_pu.at[node]

#         features_gt.update({
#             "optimization_time": optimization_time,
#             #"num_switches_switched": num_switches_switched
#         })

#         # Save optimized network and features
#         # with open(pp_gt_dir / f"{graph_id}.json", "w") as f:
#         #     json.dump(pp.to_json(updated_net), f)

#         with open(feat_gt_dir / f"{graph_id}.pkl", "wb") as f:
#             pkl.dump(features_gt, f)

#         print(f"Saved ground truth data for {graph_id}")

#     # Save optimization metrics
#     metrics_df = pd.DataFrame(metrics_data)
#     metrics_df.to_csv(folder_path / "optimization_metrics.csv", index=False)
#     print(f"\n✓ Saved optimization metrics to {folder_path / 'optimization_metrics.csv'}")

#     # Summary
#     print("\n" + "="*50)
#     print("OPTIMIZATION SUMMARY")
#     print("="*50)
#     print(f"Total graphs processed: {len(pp_networks)}")
#     print(f"Successful optimizations: {len(metrics_data)}")
#     print(f"Average optimization time: {metrics_df['optimization_time'].mean():.4f} seconds")
#     print(f"Total switches changed: {metrics_df['switches_changed'].sum()}")
#     print("="*50)

# def is_radial_and_connected(net):
#     """
#     Checks if the pandapower network `net` is radial (i.e. forms a tree)
#     and connected (i.e. one connected component).

#     Returns:
#         is_radial (bool): True if the network is a tree.
#         is_connected (bool): True if the network is fully connected.
#     """
#     G = nx.Graph()

#     # Add all buses as nodes
#     for bus in net.bus.index:
#         G.add_node(bus)

#     # Add edges for each line that is in service (all related switches closed)
#     for idx, line in net.line.iterrows():
#         add_edge = True
#         if hasattr(net, "switch") and not net.switch.empty:
#             switches = net.switch[(net.switch.et == 'l') & (net.switch.element == idx)]
#             if not switches.empty and not all(switches["closed"]):
#                 add_edge = False
#         if add_edge:
#             G.add_edge(line["from_bus"], line["to_bus"])

#     # A tree (radial network) should have exactly (n - 1) edges if connected.
#     num_edges = G.number_of_edges()
#     num_nodes = G.number_of_nodes()
#     is_radial = num_edges == num_nodes - 1
#     is_connected = nx.is_connected(G)
    
#     return is_radial, is_connected

# from pyomo.util.infeasible import log_infeasible_constraints
# from pyomo.environ import value, Constraint, Var


# def debug_infeasibility(model, tol=1e-6):
#     print("=== Infeasible Constraints ===")
#     log_infeasible_constraints(model, log_expression=True, tol=tol)

# def print_constraint_violations(model, tol=1e-6):
#     print("\n=== Constraint Violations ===")
#     for constr in model.component_data_objects(Constraint, active=True):
#         try:
#             lower = constr.lower if constr.lower is not None else -float('inf')
#             upper = constr.upper if constr.upper is not None else float('inf')
#             body_val = value(constr.body)
#             violation = max(lower - body_val, body_val - upper, 0)
#             if violation > tol:
#                 print(f"{constr.name} (index: {constr.index() if hasattr(constr, 'index') else ''}) "
#                       f"violation: {violation:.4e}, body value: {body_val:.4e}, bounds: ({lower}, {upper})")
#         except Exception as e:
#             print(f"Could not evaluate constraint {constr.name}: {e}")

# def init_logging(method):
#     # Create the data_generation directory if it does not exist
#     logging.basicConfig(level=logging.DEBUG)
#     log_dir = Path(__file__).parent / "logs"
#     log_dir.mkdir(parents=True, exist_ok=True)
    
#     # Log filename based on method (SOCP or MILP)
#     log_filename = log_dir / f"{method.upper()}_logs.txt"
    
#     # Create a logger with a unique name for your application
#     logger = logging.getLogger("network_optimizer")
#     logger.setLevel(logging.INFO)  # or INFO for less verbosity
    
#     # Remove any existing handlers
#     if logger.hasHandlers():
#         logger.handlers.clear()
    
#     # File handler
#     file_handler = logging.FileHandler(log_filename, mode="w")
#     formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
    
#     # Stream handler (console)
#     stream_handler = logging.StreamHandler(sys.stdout)
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)
    
#     logger.info(f"Logging initialized. All logs will be stored in {log_filename}")
#     return logger
    


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Generate ground truth data for power networks using optimization')
#     parser.add_argument('--folder_path', default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_val_real__range-30-150_nTest-5_nVal-5_2732025_1", type=str, help='Dataset folder path')
#     parser.add_argument('--set', type=str, choices=['test', 'validation', 'train', '', 'all'], default='test', help='Dataset set to process; leave empty for no subfolder')
#     parser.add_argument('--method', type=str, choices=['SOCP', 'MILP'], default='SOCP', help='Choose optimization method: SOCP or MILP')
#     parser.add_argument('--debug', type=bool, default=True, help='Print debug information')

#     # SOCP toggles
#     parser.add_argument('--include_voltage_drop_constraint', type=bool, default=True, help="Include voltage drop constraint SOCP")
#     parser.add_argument('--include_voltage_bounds_constraint', type=bool, default=True, help="Include voltage bounds constraint SOCP")
#     parser.add_argument('--include_power_balance_constraint', type=bool, default=True, help="Include power balance constraint SOCP")
#     parser.add_argument('--include_radiality_constraints', type=bool, default=True, help="Include radiality constraints SOCP")
#     parser.add_argument('--use_spanning_tree_radiality', type=bool, default=True, help="Use spanning tree radiality SOCP")
#     parser.add_argument('--include_switch_penalty', type=bool, default=False, help="Include switch penalty in objective SOCP")

#     # MILP toggles


#     args = parser.parse_args()

#     logger =init_logging(args.method)
    
#     SOCP_toggles = { 
#                 "include_voltage_drop_constraint": args.include_voltage_drop_constraint, 
#                 "include_voltage_bounds_constraint": args.include_voltage_bounds_constraint,   
#                 "include_power_balance_constraint": args.include_power_balance_constraint,  
#                 "include_radiality_constraints": args.include_radiality_constraints,
#                 "use_spanning_tree_radiality": args.use_spanning_tree_radiality,  
#                 "include_switch_penalty": args.include_switch_penalty,
#             }
    
#     MILP_toggles = { 
#     } 
#     if args.method == "SOCP":
#         toggles = SOCP_toggles
#     else:
#         toggles = MILP_toggles  
#     print("Toggles for optimization:")
#     print(toggles)
#     if args.set:
#         apply_optimization_and_store_ground_truths(Path(args.folder_path) / args.set, method=args.method, toggles=toggles, debug=args.debug, logger= logger)
#     elif args.set == "all": 
#         for set_name in Path(args.folder_path).iterdir():
#             if set_name.is_dir():
#                 print("\nProcessing set:", set_name)
#                 apply_optimization_and_store_ground_truths(Path(args.folder_path) / set_name, method=args.method, toggles=toggles, debug=args.debug, logger= logger)
#     else:
#         apply_optimization_and_store_ground_truths(args.folder_path, method=args.method, toggles=toggles, debug=args.debug, logger= logger)

#     print("\nGround truth generation complete!!!!")

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
from MILP_class_dnr import MILP_class  
from load_data import load_graph_data

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
        print(f"\n{'='*30} Processing {graph_id} {'='*30}")
    
        if isinstance(net, str):
            net = pp.from_json_string(net)
            print("in loop loaded network as string")

        original_switch_states = net.switch["closed"].copy() if hasattr(net, "switch") else None

        # Run a power flow to check convergence.
        pp.runpp(net, enforce_q_lims=False)
        if not net.converged:
            print(f"Power flow did not converge for {graph_id}. Skipping optimization.")
            continue

        # Instantiate optimizer based on chosen method
        if method.upper() == "MILP":
            optimizer = MILP_class(net, graph_id, logger=logger, toggles=toggles)
        else:
            optimizer = SOCP_class(net, graph_id, logger=logger, toggles=toggles)

        if hasattr(optimizer, 'initialize'):
            optimizer.initialize()  
        try:
            optimizer.initialize_with_alternative_mst(penalty=1.0)
        except Exception as e:
            print(f"Error during initialization with alternative MST for {graph_id}: {e}")
    
        model = optimizer.create_model()  # Build the optimization model

        start_time = time.time()
        results = optimizer.solve()
        optimization_time = time.time() - start_time

        # Retrieve number of switches changed (assuming optimizer.num_switches_changed is updated)
        num_switches_changed = optimizer.num_switches_changed if hasattr(optimizer, 'num_switches_changed') else 0

        print(f"Optimization solved in {optimization_time:.4f} seconds, Switches changed: {num_switches_changed}")

        metrics_data.append({
            "graph_id": graph_id,
            "optimization_time": optimization_time,
            "switches_changed": num_switches_changed
        })

        # Update features with ground truths (if needed)
        features_gt = features.get(graph_id, {}).copy()
        features_gt.update({
            "optimization_time": optimization_time,
            "switches_changed": num_switches_changed
        })

        # Save optimized features
        with open(feat_gt_dir / f"{graph_id}.pkl", "wb") as f:
            pkl.dump(features_gt, f)

        print(f"Saved ground truth data for {graph_id}")

    # Save optimization metrics
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(folder_path / "optimization_metrics.csv", index=False)
    print(f"\n✓ Saved optimization metrics to {folder_path / 'optimization_metrics.csv'}")

    # Summary printing
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Total graphs processed: {len(pp_networks)}")
    print(f"Successful optimizations: {len(metrics_data)}")
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

    Returns:
        is_radial (bool): True if the network is a tree.
        is_connected (bool): True if the network is fully connected.
    """
    G = nx.Graph()

    # Add all buses as nodes
    for bus in net.bus.index:
        G.add_node(bus)

    # Add edges for each line that is in service (all related switches closed)
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
                print(f"{constr.name} (index: {constr.index() if hasattr(constr, 'index') else ''}) "
                      f"violation: {violation:.4e}, body value: {body_val:.4e}, bounds: ({lower}, {upper})")
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
    parser.add_argument('--folder_path', default=r"data\transformed_subgraphs_27032025_4"
                        #default = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_val_real__range-30-150_nTest-10_nVal-10_2732025_3"
                        , type=str, help='Dataset folder path')
    parser.add_argument('--set', type=str, choices=['test', 'validation', 'train', '', 'all'], default='test', help='Dataset set to process; leave empty for no subfolder')
    parser.add_argument('--method', type=str, choices=['SOCP', 'MILP'], default='SOCP', help='Choose optimization method: SOCP or MILP')
    parser.add_argument('--debug', type=bool, default=True, help='Print debug information')

    # SOCP toggles
    parser.add_argument('--include_voltage_drop_constraint', type=bool, default=True, help="Include voltage drop constraint SOCP")
    parser.add_argument('--include_voltage_bounds_constraint', type=bool, default=True, help="Include voltage bounds constraint SOCP")
    parser.add_argument('--include_power_balance_constraint', type=bool, default=True, help="Include power balance constraint SOCP")
    parser.add_argument('--include_radiality_constraints', type=bool, default=True, help="Include radiality constraints SOCP")
    parser.add_argument('--use_spanning_tree_radiality', type=bool, default=True, help="Use spanning tree radiality SOCP")
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

