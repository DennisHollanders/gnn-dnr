from pathlib import Path
import geopandas as gpd
import pandas as pd
import logging
import os
import sys
import argparse
from datetime import datetime
from functools import lru_cache

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    
from electrify_subgraph import transform_subgraphs
from logger_setup import setup_logging
from create_testval import generate_flexible_datasets

def data_paths(data_dir):
    base = Path(data_dir)
    return {
        'cbs': base / "cbs_pc6_2023.gpkg",
        'buurt': base / 'buurt_to_postcodes.csv',
        'std': base / "cleaned_energ_standard_energy_data.csv"
    }

@lru_cache(maxsize=1)
def load_static_data(data_dir):
    paths = data_paths(data_dir)
    cbs = gpd.read_file(paths['cbs'])
    buurt = pd.read_csv(paths['buurt'])
    #cons = pd.read_csv(paths['cons'])
    std  = pd.read_csv(paths['std'])
    return cbs, buurt, None, std


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate electrified network datasets')
    
    # Input and output options
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='Data directory path (default: data)')
    parser.add_argument('--save_dir', type=str, default='data', 
                        help='Directory to save outputs ')
    parser.add_argument('--multiprocessing', default=False,
                        help='Enable multiprocessing for data generation')
    # Subgraph options
    parser.add_argument('--subgraph_folder', type=str, 
    #default='filtered_complete_subgraphs_final.pkl',
    
    #default = r"/vast.mnt/home/20174047/gnn-dnr/data/cbs_buurts_nodes/",
    default=r"data/cbs_buurts_small/",
                        help='Filename for subgraphs input ')
    parser.add_argument('--n_loadcase_time_intervals', type=int, default=1,
                        help='Number of time intervals per sample (default: 1)')
    parser.add_argument('--interval_duration_minutes', type=int, default=15,
                        help='Duration of each time interval in minutes (default: 15)')
    
    # Subgraph sampling options
    parser.add_argument('--iterate_all', action='store_true',
                        help='Iterate over all subgraphs')
    # if iterate_all == True:
    parser.add_argument('--n_samples_per_graph', type=int, default=1,
                        help='Number of samples per subgraph')
    # if iteral_all == False:
    parser.add_argument('--num_subgraphs', type=int, default=10,
                        help='Number of subgraphs to sample if not iterating')
    parser.add_argument('--target_busses', type=int, default=130,
                        help='Target number of busses in sampled subgraphs')
    parser.add_argument('--bus_range', type=int, default=100,
                        help='Range around target number of busses (default: 5)')
    
    # Visualization options
    parser.add_argument('--plot_subgraphs', default =False,
                        help='Plot the transformed subgraphs')
    parser.add_argument('--plot_added_edge',default=False,
                        help='Plot the added edge in subgraphs')
    parser.add_argument('--plot_distributions', action='store_true',
                        help='Plot the distributions')
    parser.add_argument('--num_plots', type=int, default=3,
                        help='Number of subgraphs to plot (default: 3)')
    parser.add_argument("--show_pandapower_report", type=bool, default=True,
                        help="Show the pandapower report for each subgraph")
    
    # Edge selection parameters
    parser.add_argument('--deterministic',default=False,
                        help='Always select the best switch edge instead of sampling')
    parser.add_argument('--top_x', type=int, default=5,
                        help='Number of top edges to consider for selection ')
    parser.add_argument('--weight_factor', type=float, default=0.92,
                        help='Weight factor for distance in edge selection ')
    parser.add_argument('--within_layers', action='store_true', default=True,
                        help='Unconstrain edge selection within layers')
    parser.add_argument("--min_distance_threshold", type=float, default=1,)
    
    # # Modification parameters
    parser.add_argument('--consumption_std', type=float, default=0.3,
                        help='Standard deviation for consumption variation (default: 0.4)')
    parser.add_argument('--production_std', type=float, default=0.2,
                        help='Standard deviation for production variation (default: 0.6)')
    parser.add_argument('--net_load_std', type=float, default=0.2,
                        help='Standard deviation for net load variation (default: 0.5)')
    parser.add_argument("--add_voltage_variation", type=bool, default=True,)
    # Save options
    parser.add_argument('--no_save', action='store_true',
                        help='Disable saving of outputs')
    parser.add_argument('--logging', default=True,
                        help='Enable logging')
    parser.add_argument('--logging_level', type=str, default='INFO',
                        help='Logging level (default: INFO)')
                        
    # Test and validation set options
    parser.add_argument('--generate_synthetic_data', default=False,
                        help='Generate training data')
    parser.add_argument('--sample_real_data', default=True,
                        help='Generate test and validation sets')
    
    parser.add_argument('--force_graph_topology', type=str, nargs='*', default=["pp_case33bw"],
                        help='List of specific graph topologies to force include (e.g., ["pp_case33bw", "simbench_1-MV-rural--0-sw"])')
    parser.add_argument('--dataset_names', type=str, nargs='+', default=["test-small", ],
                        help='Names of datasets to create (e.g. --dataset_names test validation)')
    parser.add_argument('--samples_per_dataset', type=int, nargs='+', default=[100],
                        help='Number of samples for each dataset (e.g. --samples_per_dataset 100 100)')


    parser.add_argument('--load_variation_range', type=float, nargs=2, default=[0.7, 1.3],
                        help='Range for load variation in test/val sets ')
    parser.add_argument('--bus_range_test_val', type=int, nargs=2, default=[3, 30],
                        help='Bus range for test/val sets ')
    parser.add_argument('--require_switches', action='store_true', default=True,
                        help='Require switches in test/val networks')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed for reproducibility')

    # Add new cable-based arguments
    parser.add_argument('--max_line_loading', type=float, default=120.0,
                        help='Maximum allowed line loading percentage (default: 110%)')
    parser.add_argument('--target_loading_min', type=float, default=30.0,
                        help='Minimum target line loading percentage (default: 60%)')
    parser.add_argument('--target_loading_max', type=float, default=70.0,
                        help='Maximum target line loading percentage (default: 90%)')
    parser.add_argument('--dg_penetration', type=float, default=0.2,
                        help='Distributed generation penetration rate (default: 0.2 = 20%)')
    
    return parser.parse_args()


def main():
    
    args = parse_arguments()
    # Assert that dataset_names and samples_per_dataset have equal lengths
    assert len(args.dataset_names) == len(args.samples_per_dataset), \
        f"dataset_names length ({len(args.dataset_names)}) must equal samples_per_dataset length ({len(args.samples_per_dataset)})"
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    #Initialize logging
    queue_handler, queue_listener = setup_logging(run_tag="data_generation", level_file =args.logging_level,level_console=args.logging_level)

    # Start the listener in the main process
    queue_listener.start()
    logger = logging.getLogger(__name__)  

    # timestamp & unique save path
    current_date = datetime.now()
    date_str = current_date.strftime("%d%m%Y")
    data_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.data_dir)))
    mode = "all" if args.iterate_all else f"range-{args.target_busses}-{args.bus_range}"
    jobname = os.environ.get("SLURM_JOB_NAME")
    suffix = f"{mode}" + (f"_{jobname}" if jobname else "")
    save_location = data_dir / f"transformed_subgraphs_{date_str}_{suffix}"
    counter = 1
    while save_location.exists():
        save_location = data_dir / f"transformed_subgraphs_{date_str}_{suffix}_{counter}"
        counter += 1
    
    # Define distributions
    distributions = {
        'n_switches': {'type': 'normal', 'mean': 5, 'std': 5, 'min': 3, 'max': 20, 'is_integer': True},\
        'n_busses': {'type': 'normal', 'mean': args.target_busses, 'std': 5, 'min': args.target_busses - args.bus_range, 
                    'max': args.target_busses + args.bus_range, 'is_integer': True},
        'layer_list': {'type': 'categorical', 'choices': [[0,1,2,3,4,5,6], [0,1,2,3,], [1,2,3], [0,2,3,4], [1,2],[0,1]], 
                      'weights': [0.2, 0.2, 0.2, 0.2, 0.2,0.2]},
        "standard_cables": {'type': 'categorical', 'choices': ['standard_cable_1', 'standard_cable_2', 'standard_cable_3'], 
                           'weights': [0.3, 0.3, 0.3]},
        "n_slack_busses": {'type': 'normal', 'mean': 2, 'std': 1, 'min': 1, 'max': 4, 'is_integer': True},
        "dg_penetration": {'type': 'beta', 'alpha': 2, 'beta': 5, 'min': 0.1, 'max': 0.4},
        "slack_vm_set": {"type":"uniform", "min": 1.0, "max": 1.05},
    }
    

    if args.generate_synthetic_data:
        logger.info("\nGenerating training data...")
        # load static data 
        cbs_pc6_gdf, buurt_to_postcodes, consumption_df, standard_consumption_df = load_static_data(args.data_dir)
        cbs_pc6_gdf = cbs_pc6_gdf[["geometry", "postcode6"]]
        dataframes = [consumption_df, cbs_pc6_gdf, buurt_to_postcodes, standard_consumption_df]

        # Start transformation
        logger.info("Starting transformation")
        transform_stats = transform_subgraphs(distributions, dataframes, args)

    # Generate test and validation sets if requested
    if args.sample_real_data:

        logger.info("\nGenerating test and validation sets...")
        # Generate test and validation sets
        datasets = generate_flexible_datasets(
                args, 
                dataset_names=args.dataset_names,
                samples_per_dataset=args.samples_per_dataset,
                force_topologies=args.force_graph_topology,
                max_workers=4
            )

    queue_listener.stop()
        
if __name__ == "__main__":

    main()

