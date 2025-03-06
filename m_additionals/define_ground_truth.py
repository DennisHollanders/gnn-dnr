import json
import pandas as pd
import pandapower as pp
import matplotlib.pyplot as plt
from PIL import Image
import pickle as pkl
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import os
import sys 
import time 
from pathlib import Path

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "socp_lingkang","src"))
if src_path not in sys.path:
    sys.path.append(src_path)
    
print(f"src_path: {src_path}")

from SOCP_class import SOCP_class

# Define the save location where original data is stored
data_folder = os.environ.get("DATA_FOLDER", "test_data")
SAVE_LOCATION = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))) / data_folder

def load_original_data(save_location):
    """
    Load stored NetworkX graphs, Pandapower networks, and node/edge features.
    """
    print("\n Loading stored data...")

    # Load NetworkX graphs
    with open(f"{save_location}/networkx_graphs.pkl", "rb") as f:
        nx_graphs = pkl.load(f)
    print(f" Loaded {len(nx_graphs)} NetworkX graphs.")
    with open(f"{save_location}/pandapower_networks.json", "r") as f:
        pp_networks = json.load(f)
    pp_networks = {k: pp.from_json_string(v) for k, v in pp_networks.items()}
    print(f" Loaded {len(pp_networks)} Pandapower networks.")

    # Load node and edge features
    with open(f"{save_location}/graph_features.pkl", "rb") as f:
        features = pkl.load(f)
    print(f" Loaded {len(features)} feature sets.")

    return nx_graphs, pp_networks, features

def check_bus_mapping(socp):
    missing = []
    for _, row in socp.net.line.iterrows():
        if row["from_bus"] not in socp.bus_dict:
            missing.append(row["from_bus"])
    if missing:
        print("Missing bus indices in bus_dict:", set(missing))
    else:
        print("All from_bus entries are present in bus_dict.")

def check_matrices(socp):
    print("M_f shape:", socp.M_f.shape)
    print(socp.M_f.head())
    print("M_l shape:", socp.M_l.shape)
    print(socp.M_l.head())
    print("M_w shape:", socp.M_w.shape)
    print(socp.M_w.head())


def apply_socp_and_store_ground_truths(save_location, pp_networks, features):
    print("\n Applying SOCP optimization on networks...\n")
    optimized_networks = {}
    updated_features = features.copy()
    for graph_id, net in pp_networks.items():
        print("--" * 50, "\n",
              "optimizing for net: {graph_id}     ->  ", net)
        if net.bus.empty or net.load.empty or net.gen.empty or net.line.empty:
            raise ValueError(f"⚠ Missing critical components in {graph_id}.")
        socp = SOCP_class(net, graph_id, "static", net.load, net.gen, net.sgen)
        print(f"SOCP object initialized for {graph_id}")
        socp.initialize()
        print("Initialized SOCP")
        check_bus_mapping(socp)
        socp.Pyomo_model_creation_from_pp(n_ts=[0])
        print("Pyomo model created")
        check_matrices(socp)

         # Store original switch states before optimization
        if hasattr(net, "switch") and not net.switch.empty:
            original_switch_states = net.switch["closed"].copy()
        else:
            original_switch_states = None

        start_time = time.time()
        socp.solving_SOCP_model()
        optimization_time = time.time() - start_time
        print(f"SOCP optimization solved in {optimization_time:.4f} seconds")

        # Count number of switches that changed state during optimization
        num_switches_switched = 0
        if original_switch_states is not None:
            # Assuming net.switch["closed"] is updated during optimization
            num_switches_switched = int((original_switch_states != net.switch["closed"]).sum())
            print(f"Number of switches switched: {num_switches_switched}")
        else:
            print("No switches found in network.")
            
        optimized_networks[graph_id] = pp.to_json(net)
        print(f"Stored optimized network for {graph_id}")
        for node in net.bus.index:
            if node in updated_features[graph_id]["node_features"]:
                updated_features[graph_id]["node_features"][node]["v_gt"] = net.res_bus.vm_pu.at[node]
        updated_features[graph_id]["optimization_time"] = optimization_time
        print(f"Optimization complete for {graph_id}.")
    with open(f"{save_location}/pandapower_networks_ground_truths.json", "w") as f:
        json.dump(optimized_networks, f, indent=4)
    print("\n Saved optimized Pandapower networks as 'pandapower_networks_ground_truths.json'.")
    with open(f"{save_location}/graph_features.pkl", "wb") as f:
        pkl.dump(updated_features, f)
    print(" Saved updated graph features with ground truths in 'graph_features.pkl'.")

graph_dict, pp_networks, features = load_original_data(SAVE_LOCATION)
print("graph_dict", pp_networks)
apply_socp_and_store_ground_truths(SAVE_LOCATION, pp_networks, features)