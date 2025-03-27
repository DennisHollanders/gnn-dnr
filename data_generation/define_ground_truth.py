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

# Add necessary source paths
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
load_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_search"))
sys.path.extend([src_path, load_data_path])

from SOCP_class_dnr import SOCP_class
from MILP_class_dnr import MILP_class  
from load_data import load_graph_data

def apply_optimization_and_store_ground_truths(folder_path, method="SOCP", toggles=None, debug =False):
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

        # check if the network runs correctly! 
        pp.runpp(net, enforce_q_lims=False)
        if not net.converged:
            print(f"Power flow did not converge for {graph_id}. Skipping optimization.")
            continue

        # Instantiate optimizer based on chosen method
        if method.upper() == "MILP":
            optimizer = MILP_class(net, graph_id)
        else:
            optimizer = SOCP_class(net, graph_id)

        if hasattr(optimizer, 'initialize'):
            optimizer.initialize()
        model = optimizer.create_model(toggles = toggles)

        if debug:
            debug_infeasibility(model)
            print_constraint_violations(model)
        
        start_time = time.time()
        results = optimizer.solve(model=model)
        optimization_time = time.time() - start_time

        updated_net = optimizer.update_network()

        # Calculate switch state changes
        num_switches_switched = 0
        if original_switch_states is not None:
            num_switches_switched = int((original_switch_states != updated_net.switch["closed"]).sum())
        print(f"Optimization solved in {optimization_time:.4f} seconds, Switches changed: {num_switches_switched}")

        metrics_data.append({
            "graph_id": graph_id,
            "optimization_time": optimization_time,
            "switches_changed": num_switches_switched
        })

        # Update features with ground truths
        features_gt = features.get(graph_id, {}).copy()
        node_features = features_gt.get("node_features", {})
        for node in updated_net.bus.index:
            if node in node_features:
                node_features[node]["v_gt"] = updated_net.res_bus.vm_pu.at[node]

        features_gt.update({
            "optimization_time": optimization_time,
            "num_switches_switched": num_switches_switched
        })

        # Save optimized network and features
        with open(pp_gt_dir / f"{graph_id}.json", "w") as f:
            json.dump(pp.to_json(updated_net), f)

        with open(feat_gt_dir / f"{graph_id}.pkl", "wb") as f:
            pkl.dump(features_gt, f)

        print(f"Saved ground truth data for {graph_id}")

    # Save optimization metrics
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(folder_path / "optimization_metrics.csv", index=False)
    print(f"\n✓ Saved optimization metrics to {folder_path / 'optimization_metrics.csv'}")

    # Summary
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Total graphs processed: {len(pp_networks)}")
    print(f"Successful optimizations: {len(metrics_data)}")
    print(f"Average optimization time: {metrics_df['optimization_time'].mean():.4f} seconds")
    print(f"Total switches changed: {metrics_df['switches_changed'].sum()}")
    print("="*50)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ground truth data for power networks using optimization')
    parser.add_argument('--folder_path', default=r"data\test_val_real__range-30-150_nTest-5_nVal-5_2732025_1", type=str, help='Dataset folder path')
    parser.add_argument('--set', type=str, choices=['test', 'validation', 'train', '', 'all'], default='test', help='Dataset set to process; leave empty for no subfolder')
    parser.add_argument('--method', type=str, choices=['SOCP', 'MILP'], default='SOCP', help='Choose optimization method: SOCP or MILP')
    parser.add_argument('--debug', type=bool, default=True, help='Print debug information')

    # SOCP toggles
    parser.add_argument('--include_voltage_drop_constraint', type=bool, default=True, help="Include voltage drop constraint SOCP")
    parser.add_argument('--include_voltage_bounds_constraint', type=bool, default=True, help="Include voltage bounds constraint SOCP")
    parser.add_argument('--include_power_balance_constraint', type=bool, default=True, help="Include power balance constraint SOCP")
    parser.add_argument('--include_radiality_constraints', type=bool, default=False, help="Include radiality constraints SOCP")
    parser.add_argument('--use_spanning_tree_radiality', type=bool, default=False, help="Use spanning tree radiality SOCP")
    parser.add_argument('--include_switch_penalty', type=bool, default=False, help="Include switch penalty in objective SOCP")

    # MILP toggles


    args = parser.parse_args()

    SOCP_toggles = { 
                "include_voltage_drop_constraint": args.include_voltage_drop_constraint, 
                "include_voltage_bounds_constraint": args.include_voltage_bounds_constraint,   
                "include_power_balance_constraint": args.include_power_balance_constraint,  
                "include_radiality_constraints": args.include_radiality_constraints,
                "use_spanning_tree_radiality": args.use_spanning_tree_radiality,  
                "include_switch_penalty": args.include_switch_penalty,
            }
    
    MILP_toggles = { 
    } 
    if args.method == "SOCP":
        toggles = SOCP_toggles
    else:
        toggles = MILP_toggles  



    if args.set:
        apply_optimization_and_store_ground_truths(Path(args.folder_path) / args.set, method=args.method, toggles=toggles, debug=args.debug)
    elif args.set == "all": 
        for set_name in Path(args.folder_path).iterdir():
            if set_name.is_dir():
                print("\nProcessing set:", set_name)
                apply_optimization_and_store_ground_truths(Path(args.folder_path) / set_name, method=args.method, toggles=toggles, debug=args.debug)
    else:
        apply_optimization_and_store_ground_truths(args.folder_path, method=args.method, toggles=toggles, debug=args.debug)

    print("\nGround truth generation complete!!!!")
