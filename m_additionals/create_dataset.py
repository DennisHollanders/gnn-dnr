import pickle as pkl
from pathlib import Path
import geopandas as gpd
import pandas as pd
import logging 
import os
import sys
from datetime import datetime

# Get the path to the 'src' folder relative to the notebook
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)
    
from electrify_subgraph2 import transform_subgraphs
from logger_setup import logger 

date_str = datetime.now().strftime("%d%m%Y")
save_location = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))) / f"transformed_subgraphs_{date_str}"
counter = 1
while save_location.exists():
    save_location = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))) / f"transformed_subgraphs_{date_str}_{counter}"
    counter += 1

# abs path
path_to_graphs = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "filtered_complete_subgraphs.pkl")))
with open(path_to_graphs, 'rb') as f:
    subgraphs = pkl.load(f)

print("amount of subgraphs opened:", len(subgraphs))

# Load datasets
cbs_pc6_gpkg =Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))) / "cbs_pc6_2023.gpkg"
buurt_to_postcodes_csv = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))) /'buurt_to_postcodes.csv'
consumption_df_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))) /  "aggregated_kleinverbruik_with_opwek.csv"
standard_consumption_df_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))) / "cleaned_energ_standard_energy_data.csv"

cbs_pc6_gdf = gpd.read_file(cbs_pc6_gpkg.resolve())
buurt_to_postcodes = pd.read_csv(buurt_to_postcodes_csv.resolve())
consumption_df = pd.read_csv(consumption_df_path.resolve()) 
standard_consumption_df = pd.read_csv(standard_consumption_df_path.resolve())

cbs_pc6_gdf = cbs_pc6_gdf[["geometry","postcode6"]]
dfs = [consumption_df, cbs_pc6_gdf, buurt_to_postcodes, standard_consumption_df]

distributions = {
    'n_switches': {'type': 'normal', 'mean': 2, 'std':1,  'min': 1, 'max': 5,  'is_integer': True},
    "num_substations": {'type': 'normal', 'mean': 1, 'std':1,  'min': 1, 'max': 3,  'is_integer': True},
    'n_busses': {'type': 'normal', 'mean': 25, 'std': 10, 'min': 10, 'max': 200,  'is_integer': True},
    'layer_list': {'type': 'categorical', 'choices': [[0,1,2,3,4,5]], 'weights': [1]},
    "split": {'type': 'categorical', 'choices': ['load', 'gen', 'transfer'], 'weights': [0.9, 0.1, 0.0]},
    "standard_cables": {'type': 'categorical', 'choices': ['standard_cable_1', 'standard_cable_2', 'standard_cable_3'], 'weights': [0.3, 0.3, 0.3]},

}
kwargs = {
    'PV': True,                     # If True, add PV to the graph
    "n_loadcase_time_intervals": 1,  # One day (96 intervals of 15 minutes)
    "n_samples_per_graph": 3,  # Three different samples per subgraph

    # Subgraph Sampling options 
    'is_iterate': True,            # If True, iterate over all subgraphs.
    'amount_of_subgraphs':3,       # Amount of subgraphs to sample if is_iterate is False.
    "plot_added_edge": True,
    'plot_subgraphs': False,         # If True, plot the transformed subgraphs.
    'plot_distributions': False,     # If True, plot the distributions.
    'amount_to_plot': 3,            # Amount of subgraphs to plot.

    # Hyperparameters for edge selection
    'deterministic': False,         # If True, always select the best switch edge. If False, sample from top_x.
    'top_x': 5,                     # Amount of top edges to consider
    'weight_factor': 0.9,           # Weight factor for distance in edge selection.  1 = only cycle length, 0 = only distance
    "within_layers": True,

    "modify_subgraph_each_sample": True,
    "consumption_std": 0.4,
    "production_std": 0.6,
    "net_load_std": 0.5,
    "interval_duration_minutes": 15,
    "save": True,
    "save_location": save_location, 
    "logging": False,
}

print(f"graphs to be generated: {len(subgraphs) * kwargs['n_samples_per_graph'] * kwargs['n_loadcase_time_intervals']}")
logger.info(f"Graphs to be generated: {len(subgraphs) * kwargs['n_samples_per_graph'] * kwargs['n_loadcase_time_intervals']}")

print("Starting transformation")
new_subgraphs = transform_subgraphs(subgraphs, distributions,dfs,kwargs, logger)