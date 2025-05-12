import os
import glob
import time
import json
import pickle as pkl
import copy
import random
import numpy as np
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import pandapower as pp
import pandapower.networks as pn
import simbench as sb
import pandapower.topology as top
import logging

logger = logging.getLogger(__name__)

# Create a cache directory for networks
CACHE_DIR = Path("network_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Global registry for SimBench timepoints
network_timepoints: Dict[str, Dict[str, Any]] = {}

def has_switches(net):
    return hasattr(net, 'switch') and not net.switch.empty and len(net.switch) > 0


def load_and_cache_network(network_id, cache_dir=CACHE_DIR):
    # Create cache file path
    cache_file = cache_dir / f"{network_id}.pkl"
    
    # Check if network is in cache
    if cache_file.exists():
        try:
            start_time = time.time()
            with open(cache_file, 'rb') as f:
                net = pkl.load(f)
            print(f"Loaded {network_id} from cache in {time.time() - start_time:.2f}s")
            return net
        except Exception as e:
            print(f"Error loading cached network {network_id}: {e}")
            # Continue to load from source if cache loading fails
    
    # Load from source
    try:
        start_time = time.time()
        if network_id.startswith('simbench_'):
            code = network_id[9:]  
            net = sb.get_simbench_net(code)
        elif network_id.startswith('pp_'):
            case = network_id[3:] 
            if case == "caseIEEE30":
                net = pn.case_IEEE30() if hasattr(pn, "case_IEEE30") else pn.case30()
            else:
                net = getattr(pn, case)()
        else:
            raise ValueError(f"Unknown network type: {network_id}")
            
        # Cache the network for future use
        with open(cache_file, 'wb') as f:
            pkl.dump(net, f)
        
        print(f"Loaded and cached {network_id} in {time.time() - start_time:.2f}s")
        return net
    except Exception as e:
        print(f"Error loading network {network_id}: {e}")
        return None
    

def check_network_suitability(network_id, bus_range=(25,50), require_switches=True):
    net = load_and_cache_network(network_id)
    
    if net is None:
        return False, None
    
    # Check bus count
    if not (bus_range[0] <= len(net.bus) <= bus_range[1]):
        return False, None
    
    # Check for switches if required
    if require_switches and not has_switches(net):
        return False, None
    
    return True, net

def get_candidate_networks(bus_range=(25,50), require_switches=True, max_workers=4):
    start_time = time.time()
    candidate_networks = {}
    networks_without_switches = []
    
    # Get Simbench network codes
    list_of_codes = sb.collect_all_simbench_codes(mv_level="MV")
    mv_codes = [code for code in list_of_codes if "MV" in code]
    
    # Filter Simbench codes if require_switches
    if require_switches:
        mv_sw_codes = [code for code in mv_codes if "no_sw" not in code]
    else:
        mv_sw_codes = mv_codes
    
    # Get PandaPower network names
    standard_cases = [
        "case4gs", "case5", "case6ww", "case9", "case14", "case30", 
        "caseIEEE30", "case33bw", "case39", "case57", "case89pegase", 
        "case118", "case145"
    ]
    
    # Create network IDs for all potential candidates
    simbench_ids = [f"simbench_{code}" for code in mv_sw_codes]
    pp_ids = [f"pp_{case}" for case in standard_cases]
    all_network_ids = simbench_ids + pp_ids
    
    print(f"Checking {len(all_network_ids)} potential networks...")
    
    # Use parallel processing to check network suitability
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary of futures mapped to network IDs
        future_to_id = {
            executor.submit(check_network_suitability, 
                           network_id, 
                           bus_range, 
                           require_switches): network_id 
            for network_id in all_network_ids
        }
        
        for future in concurrent.futures.as_completed(future_to_id):
            network_id = future_to_id[future]
            try:
                suitable, net = future.result()
                if suitable and net is not None:
                    bus_count = len(net.bus)
                    switch_count = len(net.switch) if hasattr(net, 'switch') else 0
                    candidate_networks[network_id] = net
                    print(f"Added network {network_id} with {bus_count} buses and {switch_count} switches")
                elif require_switches and net is not None and not has_switches(net):
                    networks_without_switches.append(network_id)
            except Exception as e:
                print(f"Error processing {network_id}: {e}")
    
    print(f"\nFound {len(candidate_networks)} valid networks matching criteria in {time.time() - start_time:.2f}s")
    if require_switches and networks_without_switches:
        print(f"Skipped {len(networks_without_switches)} networks without switches")
    
    return candidate_networks

def apply_random_load_variations(net, load_variation_range=(0.5, 1.51)):
    """Apply random load variations to a network"""
    # Create a deep copy to avoid modifying the original
    net_case = copy.deepcopy(net)
    
    # Apply random load variations
    for idx in net_case.load.index:
        factor = np.random.uniform(load_variation_range[0], load_variation_range[1])
        net_case.load.at[idx, "p_mw"] *= factor
        net_case.load.at[idx, "q_mvar"] *= factor
    
    return net_case

network_timepoints = {}

def apply_profile_timepoint(net, profiles, time_step):
    for elm_param in profiles.keys():
        if profiles[elm_param].shape[1]:  # Check if there's any data
            elm = elm_param[0]  # Element type (e.g., 'load', 'sgen')
            param = elm_param[1]  # Parameter name (e.g., 'p_mw', 'q_mvar')
            net[elm].loc[:, param] = profiles[elm_param].loc[time_step]
    return net


def generate_combined_dataset(
    args,
    max_workers: int = 4
) -> Tuple[Dict[str,Any], Dict[str,Any]]:
    """Generate test/validation datasets, save them, and return them."""
    start = time.time()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    candidates = get_candidate_networks(
        bus_range= args.bus_range_test_val,
        require_switches=args.require_switches,
        max_workers=max_workers
    )
    if not candidates:
        raise ValueError("No suitable networks found.")

    def allocate(total: int, tag: str):
        counts = {k: 0 for k in candidates}
        ds: Dict[str,Any] = {}
        while len(ds) < total:
            for key, net in candidates.items():
                if len(ds) >= total:
                    break
                idx = counts[key]
                name, data = create_network_case(key, net, tag, idx, args.load_variation_range)
                if name:
                    ds[name] = data
                    counts[key] += 1
        return ds

    test_ds = allocate(args.test_cases, 'test')
    val_ds  = allocate(args.val_cases,  'val')

    logger.info(f"Generated {len(test_ds)} test and {len(val_ds)} validation in {time.time()-start:.1f}s")

    # versioned path
    now = datetime.now()
    min_bus_range = args.bus_range_test_val[0]
    max_bus_range = args.bus_range_test_val[1]
    bstr = f"{min_bus_range}-{max_bus_range}"
    base_name = f"test_val_real__range-{bstr}_nTest-{args.test_cases}_nVal-{args.val_cases}_{now.day}{now.month}{now.year}"
    base_abs = os.path.abspath(args.data_dir)
    # find sequence
    pattern = os.path.join(base_abs, base_name + '_*')
    existing = glob.glob(pattern)
    seq = 1
    if existing:
        nums = [int(os.path.basename(d).split('_')[-1]) for d in existing if d.split('_')[-1].isdigit()]
        if nums:
            seq = max(nums) + 1
    save_path = os.path.join(base_abs, f"{base_name}_{seq}")

    save_dataset( test_ds, val_ds,args,base_path=save_path )

    return test_ds, val_ds

def save_test_data(test_dataset, val_dataset, args):  
    # Format bus range as a string
    current_date = datetime.now()
    bus_range_str = f"{args.bus_range_test_val[0]}-{args.bus_range_test_val[1]}"
    
    day, month, year = current_date.day, current_date.month, current_date.year
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



def create_network_case(network_id, net, case_type, case_idx, load_variation_range=(0.5, 1.51)):
    """Create a single network case using SimBench profiles when available"""
    global network_timepoints
    
    try:
        # Create a deep copy to avoid modifying the original
        net_case = copy.deepcopy(net)
        used_profile = False
        selected_timepoint = None
        
        # Check if this is a SimBench network that might have profiles
        if network_id.startswith('simbench_'):
            try:
                
                # Try to get profiles from the network - this is the correct approach based on the notebook
                try:
                    # Get all available profiles for the network
                    profiles = sb.get_absolute_values(net_case, profiles_instead_of_study_cases=True)
                    
                    # Check if we've initialized timepoints for this network
                    if network_id not in network_timepoints:
                        # Determine available timepoints - these are the indices of the profiles
                        available_timesteps = list(range(len(profiles[list(profiles.keys())[0]])))
                        network_timepoints[network_id] = {
                            'timepoints': available_timesteps,
                            'used': set()
                        }
                        print(f"Found {len(available_timesteps)} timepoints for {network_id}")
                    
                    # Get unused timepoints
                    available = [tp for tp in network_timepoints[network_id]['timepoints'] 
                                if tp not in network_timepoints[network_id]['used']]
                    
                    if available:
                        # Select a random unused timepoint
                        selected_timepoint = random.choice(available)
                        network_timepoints[network_id]['used'].add(selected_timepoint)
                        
                        # Apply the timepoint to the network - this is the correct way
                        apply_profile_timepoint(net_case, profiles, selected_timepoint)
                        
                        print(f"Successfully applied SimBench profile timepoint {selected_timepoint} to {network_id}")
                        used_profile = True
                    else:
                        print(f"All timepoints used for {network_id}, resetting timepoint tracking")
                        # Reset used timepoints to allow reuse
                        network_timepoints[network_id]['used'] = set()
                        
                        # Try again with a timepoint
                        selected_timepoint = random.choice(network_timepoints[network_id]['timepoints'])
                        network_timepoints[network_id]['used'].add(selected_timepoint)
                        
                        apply_profile_timepoint(net_case, profiles, selected_timepoint)
                        print(f"Applied reused timepoint {selected_timepoint} to {network_id}")
                        used_profile = True
                        
                except Exception as e:
                    print(f"Error accessing SimBench profiles: {e}")
                    # Fall back to random variations if profiles don't work
                    net_case = apply_random_load_variations(net_case, load_variation_range)
            except Exception as e:
                print(f"Error in SimBench processing: {e}")
                net_case = apply_random_load_variations(net_case, load_variation_range)
        else:
            # For non-SimBench networks, always use random variations
            net_case = apply_random_load_variations(net_case, load_variation_range)

        nx_graph = top.create_nxgraph(net_case, respect_switches=True)
        node_count = nx_graph.number_of_nodes()

        # Create case name - include timepoint in name if one was used
        if selected_timepoint is not None and used_profile:
            case_name = f"{network_id}_{case_type}_{case_idx}_ts{selected_timepoint}_n{node_count}"
        else:
            case_name = f"{network_id}_{case_type}_{case_idx}_n{node_count}"


        switch_count = len(net_case.switch) if hasattr(net_case, 'switch') else 0
        print(f"Added {case_type} case {case_name} with {switch_count} switches")
        
        return case_name, {"network": net_case, "nx_graph": nx_graph}
    except Exception as e:
        print(f"Failed to create case for {network_id}: {e}")
        return None, None

def reset_timepoints():
    """Reset the global registry of timepoints"""
    global network_timepoints
    network_timepoints = {}

def extract_node_features2(net, nx_graph):
    """Extract node features from the power grid network."""
    nx_to_pp_bus_map = net.get("nx_to_pp_bus_map", {node: idx for idx, node in enumerate(net.bus.index)})
    node_features = {}
    for node in nx_graph.nodes:
        pp_bus_idx = nx_to_pp_bus_map.get(node, node)
        node_features[node] = {
            "p": net.res_bus.p_mw.at[pp_bus_idx] if pp_bus_idx in net.res_bus.index else None,
            "q": net.res_bus.q_mvar.at[pp_bus_idx] if pp_bus_idx in net.res_bus.index else None,
            "v": net.res_bus.vm_pu.at[pp_bus_idx] if pp_bus_idx in net.res_bus.index else None,
            "theta": net.res_bus.va_degree.at[pp_bus_idx] if pp_bus_idx in net.res_bus.index else None
        }
    return node_features

def extract_edge_features2(net, nx_graph):
    """Extract edge features from the power grid network."""
    nx_to_pp_bus_map = net.get("nx_to_pp_bus_map", {node: idx for idx, node in enumerate(net.bus.index)})
    edge_features = {}
    for idx, edge in enumerate(nx_graph.edges):
        u, v = edge[:2]
        pp_from_bus = nx_to_pp_bus_map.get(u, u)
        pp_to_bus = nx_to_pp_bus_map.get(v, v)
        matching_lines = net.line[(net.line.from_bus == pp_from_bus) & (net.line.to_bus == pp_to_bus)]
        if matching_lines.empty:
            continue
        line_idx = matching_lines.index[0]
        R = matching_lines.r_ohm_per_km.iloc[0]
        X = matching_lines.x_ohm_per_km.iloc[0]
        switch_status = int(net.switch[(net.switch.bus == pp_from_bus) & (net.switch.element == pp_to_bus)].closed.any())
        edge_features[(u, v)] = {
            "edge_idx": idx,
            "line_idx": line_idx,
            "R": R,
            "X": X,
            "switch_state": switch_status,
        }
    return edge_features

def save_combined_data(dataset, set_name, base_dir):
    """Save combined dataset to disk."""
    # Create all required directories


    nx_dir =  base_dir /set_name / "networkx_graphs"
    pp_dir = base_dir / set_name / "pandapower_networks"
    feat_dir = base_dir / set_name / "graph_features"
    
    for directory in [nx_dir, pp_dir, feat_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Process each case separately
    for case_name, data in dataset.items():
        net = data["network"]
        nx_graph = data["nx_graph"]

        # Run power flow analysis
        try:
            pp.runpp(net, max_iteration=100, v_debug=False, run_control=True, 
                    initialization="dc", calculate_voltage_angles=True)
        except Exception as e:
            print(f"Power flow did not converge for {case_name}: {e}")
            
        # Extract features
        node_feats = extract_node_features2(net, nx_graph) if nx_graph is not None else None
        edge_feats = extract_edge_features2(net, nx_graph) if nx_graph is not None else None
        features = {"node_features": node_feats, "edge_features": edge_feats}
            
        # Save each graph separately
        # 1. NetworkX graph
        nx_file = os.path.join(nx_dir, f"{case_name}.pkl")
        with open(nx_file, "wb") as f:
            pkl.dump(nx_graph, f)
            
        # 2. PandaPower network
        pp_file = os.path.join(pp_dir, f"{case_name}.json")
        with open(pp_file, "w") as f:
            json.dump(pp.to_json(net), f)
            
        # 3. Features
        feat_file = os.path.join(feat_dir, f"{case_name}.pkl")
        with open(feat_file, "wb") as f:
            pkl.dump(features, f)
    
    print(f"Saved {len(dataset)} {set_name} cases individually to {base_dir}")

def save_dataset(test_ds: Dict[str,Any],val_ds: Dict[str,Any],args, base_path: str,) -> str:
    now = datetime.now()
    day, month, year = now.day, now.month, now.year
    min_bus_range = args.bus_range_test_val[0]
    max_bus_range = args.bus_range_test_val[1]
    bstr = f"{min_bus_range}-{max_bus_range}"
    #base_name = f"test_val_real__range-{bstr}_nTest-{args.test_cases}_nVal-{args.val_cases}_{day}{month}{year}"
    base_path = Path(base_path) 
    os.makedirs(base_path, exist_ok=True)
    with open(os.path.join(base_path, 'metadata.txt'), 'w') as f:
        f.write(f"Created: {now}\nBus range: {args.bus_range}\nTest: {len(test_ds)}/{args.test_cases}\nVal: {len(val_ds)}/{args.val_cases}\n")
    save_combined_data(test_ds, 'test', base_path)
    save_combined_data(val_ds, 'validation', base_path)
    logger.info(f"Dataset saved at {base_path}")
    return base_path