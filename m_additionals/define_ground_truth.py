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

# Define the save location where original data is stored
SAVE_LOCATION = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), "data", "...")  
socp_src_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), "socp_lingkang", "src") 

sys.path.append(socp_src_path)

# Verify it was added correctly
print("Added to sys.path:", socp_src_path)

# Now you can import the SOCP class
from SOCP_class import SOCP_class

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
    """
    Apply SOCP optimization on each Pandapower network and store results.
    """
    print("\n Applying SOCP optimization on networks...\n")
    
    optimized_networks = {}  # Store optimized networks
    updated_features = features.copy()  # Copy features to update ground truth

    for graph_id, net in pp_networks.items():
        print("--"*50, "\n",
            "optimizing for net: {graph_id}     ->  ",net)

        # Ensure no missing tables
        if net.bus.empty or net.load.empty or net.gen.empty or net.line.empty:
            raise ValueError(f"⚠ Missing critical components in {graph_id}.")
        
        # Initialize SOCP optimization
        socp = SOCP_class(net, graph_id, "static", net.load, net.gen, net.sgen)
        print(f"SOCP object initialized for {graph_id}")

        socp.initialize()
        print("Initialized SOCP")
        
        check_bus_mapping(socp)
        
        socp.Pyomo_model_creation_from_pp(n_ts=[0])  # Static mode time step
        print("Pyomo model created")
        
        check_matrices(socp)

        socp.solving_SOCP_model()
        print("SOCP optimization solved")

        # Store optimized network
        optimized_networks[graph_id] = pp.to_json(net)
        print(f"Stored optimized network for {graph_id}")

        # Store optimized ground truth voltage magnitudes in features
        for node in net.bus.index:
            if node in updated_features[graph_id]["node_features"]:
                updated_features[graph_id]["node_features"][node]["v_gt"] = net.res_bus.vm_pu.at[node]

        print(f"Optimization complete for {graph_id}.")


    # Save optimized Pandapower networks as JSON
    with open(f"{save_location}/pandapower_networks_ground_truths.json", "w") as f:
        json.dump(optimized_networks, f, indent=4)
    print("\n Saved optimized Pandapower networks as 'pandapower_networks_ground_truths.json'.")

    # Save updated node and edge features with ground truth
    with open(f"{save_location}/graph_features.pkl", "wb") as f:
        pkl.dump(updated_features, f)
    print(" Saved updated graph features with ground truths in 'graph_features.pkl'.")




graph_dict, pp_networks, features = load_original_data(SAVE_LOCATION)

print("graph_dict",pp_networks,)

apply_socp_and_store_ground_truths(SAVE_LOCATION, pp_networks, features)