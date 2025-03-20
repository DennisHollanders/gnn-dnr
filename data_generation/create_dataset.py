import pickle as pkl
from pathlib import Path
import geopandas as gpd
import pandas as pd
import logging 
import os
import sys
import argparse
from datetime import datetime
import glob
import numpy as np
import random

# Get the path to the 'src' folder relative to the script
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    
from electrify_subgraph3 import transform_subgraphs
from logger_setup import logger 


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate electrified network datasets')
    
    # Input and output options
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='Data directory path (default: data)')
    parser.add_argument('--save_dir', type=str, default=None, 
                        help='Directory to save outputs (default: auto-generated)')
    
    # Subgraph options
    parser.add_argument('--subgraph_file', type=str, default='filtered_subgraphs.pkl',
                        help='Filename for subgraphs input ')
    parser.add_argument('--n_loadcase_time_intervals', type=int, default=1,
                        help='Number of time intervals per sample (default: 1)')
    parser.add_argument('--interval_duration_minutes', type=int, default=15,
                        help='Duration of each time interval in minutes (default: 15)')
    
    # Subgraph sampling options
    parser.add_argument('--iterate_all', action='store_true',default=True,
                        help='Iterate over all subgraphs')
    # if iterate_all == True:
    parser.add_argument('--n_samples_per_graph', type=int, default=5,
                        help='Number of samples per subgraph')
    # if iteral_all == False:
    parser.add_argument('--num_subgraphs', type=int, default=3,
                        help='Number of subgraphs to sample if not iterating')
    parser.add_argument('--target_busses', type=int, default=35,
                        help='Target number of busses in sampled subgraphs')
    parser.add_argument('--bus_range', type=int, default=5,
                        help='Range around target number of busses (default: 5)')
    
    # Visualization options
    parser.add_argument('--plot_subgraphs', action='store_true',
                        help='Plot the transformed subgraphs')
    parser.add_argument('--plot_added_edge', action='store_true',default=False,
                        help='Plot the added edge in subgraphs')
    parser.add_argument('--plot_distributions', action='store_true',
                        help='Plot the distributions')
    parser.add_argument('--num_plots', type=int, default=3,
                        help='Number of subgraphs to plot (default: 3)')
    parser.add_argument("--show_pandapower_report", type=bool, default=False,
                        help="Show the pandapower report for each subgraph")
    
    # Edge selection parameters
    parser.add_argument('--deterministic',default=False,
                        help='Always select the best switch edge instead of sampling')
    parser.add_argument('--top_x', type=int, default=5,
                        help='Number of top edges to consider for selection ')
    parser.add_argument('--weight_factor', type=float, default=0.75,
                        help='Weight factor for distance in edge selection ')
    parser.add_argument('--within_layers', action='store_true', default=True,
                        help='Unconstrain edge selection within layers')
    
    # Modification parameters
    parser.add_argument('--modify_each_sample', action='store_true', default=True,
                        help='Modify subgraph for each sample')
    parser.add_argument('--consumption_std', type=float, default=0.4,
                        help='Standard deviation for consumption variation (default: 0.4)')
    parser.add_argument('--production_std', type=float, default=0.6,
                        help='Standard deviation for production variation (default: 0.6)')
    parser.add_argument('--net_load_std', type=float, default=0.5,
                        help='Standard deviation for net load variation (default: 0.5)')
    
    # Save options
    parser.add_argument('--no_save', action='store_true',
                        help='Disable saving of outputs')
    parser.add_argument('--logging', action='store_true', default=True,
                        help='Enable logging')
                        
    # Test and validation set options
    parser.add_argument('--generate_test_val', default=False, action='store_true',
                        help='Generate test and validation sets')
    parser.add_argument('--test_cases', type=int, default=5,
                        help='Number of test cases to generate')
    parser.add_argument('--val_cases', type=int, default=5,
                        help='Number of validation cases to generate')
    parser.add_argument('--load_variation_range', type=float, nargs=2, default=[0.8, 1.2],
                        help='Range for load variation in test/val sets ')
    parser.add_argument('--bus_range_test_val', type=int, nargs=2, default=[30, 150],
                        help='Bus range for test/val sets ')
    parser.add_argument('--require_switches', action='store_true', default=True,
                        help='Require switches in test/val networks')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Create timestamp for saving
    current_date = datetime.now()
    date_str = current_date.strftime("%d%m%Y")

    data_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.data_dir)))

    save_location = data_dir / f"transformed_subgraphs_{date_str}"
    counter = 1
    while save_location.exists():
        save_location = data_dir / f"transformed_subgraphs_{date_str}_{counter}"
        counter += 1
    
    # Load subgraphs
    path_to_graphs = data_dir / args.subgraph_file
    with open(path_to_graphs, 'rb') as f:
        subgraphs = pkl.load(f)
    
    print(f"Number of subgraphs loaded: {len(subgraphs)}")
    
    # Load datasets
    cbs_pc6_gpkg = data_dir / "cbs_pc6_2023.gpkg"
    buurt_to_postcodes_csv = data_dir / 'buurt_to_postcodes.csv'
    consumption_df_path = data_dir / "aggregated_kleinverbruik_with_opwek.csv"
    standard_consumption_df_path = data_dir / "cleaned_energ_standard_energy_data.csv"
    
    cbs_pc6_gdf = gpd.read_file(cbs_pc6_gpkg.resolve())
    buurt_to_postcodes = pd.read_csv(buurt_to_postcodes_csv.resolve())
    consumption_df = pd.read_csv(consumption_df_path.resolve()) 
    standard_consumption_df = pd.read_csv(standard_consumption_df_path.resolve())
    
    cbs_pc6_gdf = cbs_pc6_gdf[["geometry", "postcode6"]]
    dataframes = [consumption_df, cbs_pc6_gdf, buurt_to_postcodes, standard_consumption_df]
    
    # Define distributions
    distributions = {
        'n_switches': {'type': 'normal', 'mean': 2, 'std': 1, 'min': 1, 'max': 5, 'is_integer': True},
        'n_busses': {'type': 'normal', 'mean': args.target_busses, 'std': 5, 'min': args.target_busses - args.bus_range, 
                    'max': args.target_busses + args.bus_range, 'is_integer': True},
        'layer_list': {'type': 'categorical', 'choices': [[0,1,2,3,4,5,6], [0,1,2,3,], [1,2,3], [0,2,3,4], [1,2],[0,1]], 
                      'weights': [0.2, 0.2, 0.2, 0.2, 0.2,0.2]},
        "standard_cables": {'type': 'categorical', 'choices': ['standard_cable_1', 'standard_cable_2', 'standard_cable_3'], 
                           'weights': [0.3, 0.3, 0.3]},
        "n_slack_busses": {'type': 'normal', 'mean': 2, 'std': 1, 'min': 1, 'max': 4, 'is_integer': True},
    }
    
    # Configuration parameters
    config = {
        "n_loadcase_time_intervals": args.n_loadcase_time_intervals,
        "n_samples_per_graph": args.n_samples_per_graph,
        
        # Subgraph Sampling options 
        'is_iterate': args.iterate_all,
        'amount_of_subgraphs': args.num_subgraphs,
        "plot_added_edge": args.plot_added_edge,
        'plot_subgraphs': args.plot_subgraphs,
        'plot_distributions': args.plot_distributions,
        'amount_to_plot': args.num_plots,
        "range": args.bus_range,
        
        # Hyperparameters for edge selection
        'deterministic': args.deterministic,
        'top_x': args.top_x,
        'weight_factor': args.weight_factor,
        "within_layers": args.within_layers,
        
        "modify_subgraph_each_sample": args.modify_each_sample,
        "consumption_std": args.consumption_std,
        "production_std": args.production_std,
        "net_load_std": args.net_load_std,
        "interval_duration_minutes": args.interval_duration_minutes,
        "save": not args.no_save,
        "save_location": save_location, 
        "logging": args.logging,
        "show_pandapower_report": args.show_pandapower_report,
    }
    
    # Log configuration
    total_graphs = len(subgraphs) * config['n_samples_per_graph'] * config['n_loadcase_time_intervals']
    print(f"Total graphs to be generated: {total_graphs}")
    logger.info(f"Total graphs to be generated: {total_graphs}")
    
    # Start transformation
    print("Starting transformation")
    transform_stats = transform_subgraphs(subgraphs, distributions, dataframes, config, logger)
    
    # Generate test and validation sets if requested
    if args.generate_test_val:

        print("\nGenerating test and validation sets...")
        from electrify_subgraph3 import generate_combined_dataset, save_combined_data
        
        # Generate test and validation sets
        test_dataset, val_dataset = generate_combined_dataset(
            bus_range=args.bus_range_test_val,
            test_total_cases=args.test_cases,
            val_total_cases=args.val_cases,
            load_variation_range=args.load_variation_range,
            random_seed=args.random_seed,
            require_switches=args.require_switches
        )
        
        # Format bus range as a string
        bus_range_str = f"{args.bus_range_test_val[0]}-{args.bus_range_test_val[1]}"
        
        day, month, year = date_str.day, date_str.month, date_str.year
        base_name = f"test_val_real__range-{bus_range_str}_nTest-{args.test_cases}_nVal-{args.val_cases}_{day}{month}{year}"
        
        # Find existing datasets with the same pattern to determine sequence number
        search_pattern = f"{base_name}_*"
        base_path = os.path.abspath(args.data_dir)
        existing_dirs = glob.glob(os.path.join(base_path, search_pattern))
        
        if not existing_dirs:
            sequence_num = 1
        else:
            seq_nums = []
            for dir_path in existing_dirs:
                try:
                    dir_name = os.path.basename(dir_path)
                    seq_num = int(dir_name.split('_')[-1])
                    seq_nums.append(seq_num)
                except (ValueError, IndexError):
                    continue
            sequence_num = max(seq_nums) + 1 if seq_nums else 1
        
        # Create the directory name with sequence number
        dataset_dir = f"{base_name}_{sequence_num}"
        test_val_save_location = os.path.join(base_path, dataset_dir)
        
        print(f"Creating test/validation dataset at: {test_val_save_location}")
        print(f"Test set size: {len(test_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        
        # Save datasets
        save_combined_data(test_dataset, "test", test_val_save_location)
        save_combined_data(val_dataset, "validation", test_val_save_location)
        
        print(f"Test and validation datasets saved successfully at {test_val_save_location}")
    
    return transform_stats


if __name__ == "__main__":
    main()