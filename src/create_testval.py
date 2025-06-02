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
from typing import Dict, Any, Tuple, List
import pandapower as pp
import pandapower.networks as pn
import simbench as sb
import pandapower.topology as top
import logging
import networkx as nx

logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.WARNING)

# Global registry for SimBench timepoints
network_timepoints: Dict[str, Dict[str, Any]] = {}

def has_switches(net):
    return hasattr(net, 'switch') and not net.switch.empty and len(net.switch) > 0

def load_and_cache_network(network_id):

    cache_file = Path("data_generation") / "network_cache" / f"{network_id}.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        try:
            start_time = time.time()
            with open(cache_file, 'rb') as f:
                net = pkl.load(f)
            print(f"Loaded {network_id} from cache in {time.time() - start_time:.2f}s")
            return net
        except Exception as e:
            print(f"Error loading cached network {network_id}: {e}")
    
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

    if not (bus_range[0] <= len(net.bus) <= bus_range[1]):
        return False, None

    if require_switches and not has_switches(net):
        return False, None
    
    return True, net

def get_candidate_networks(bus_range=(25,50), require_switches=True, max_workers=4):
    start_time = time.time()
    candidate_networks = {}
    networks_without_switches = []

    list_of_codes = sb.collect_all_simbench_codes(mv_level="MV")
    mv_codes = [code for code in list_of_codes if "MV" in code]

    if require_switches:
        mv_sw_codes = [code for code in mv_codes if "no_sw" not in code]
    else:
        mv_sw_codes = mv_codes
   
    standard_cases = [
        "case4gs", "case5", "case6ww", "case9", "case14", "case30", 
        "caseIEEE30", "case33bw", "case39", "case57", "case89pegase", 
        "case118", "case145", "case300", "iceland"
    ]
    
    # Create network IDs for all potential candidates
    simbench_ids = [f"simbench_{code}" for code in mv_sw_codes]
    pp_ids = [f"pp_{case}" for case in standard_cases]
    all_network_ids = simbench_ids + pp_ids
    
    logger.info(f"Checking {len(all_network_ids)} potential networks...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                    logger.info(f"Added network {network_id} with {bus_count} buses and {switch_count} switches")
                elif require_switches and net is not None and not has_switches(net):
                    networks_without_switches.append(network_id)
            except Exception as e:
                logger.info(f"Error processing {network_id}: {e}")
    
    logger.info(f"\nFound {len(candidate_networks)} valid networks matching criteria in {time.time() - start_time:.2f}s")
    if require_switches and networks_without_switches:
        logger.info(f"Skipped {len(networks_without_switches)} networks without switches")
    
    return candidate_networks

def apply_random_load_variations(net, load_variation_range=(0.5, 1.51)):
    """Apply random load variations to a network"""

    net_case = copy.deepcopy(net)

    for idx in net_case.load.index:
        factor = np.random.uniform(load_variation_range[0], load_variation_range[1])
        net_case.load.at[idx, "p_mw"] *= factor
        net_case.load.at[idx, "q_mvar"] *= factor
    
    return net_case

network_timepoints = {}

def apply_profile_timepoint(net, profiles, time_step):
    for elm_param in profiles.keys():
        if profiles[elm_param].shape[1]:
            elm = elm_param[0]  
            param = elm_param[1] 
            net[elm].loc[:, param] = profiles[elm_param].loc[time_step]
    return net


def generate_flexible_datasets(args,dataset_names: List[str],samples_per_dataset: List[int],
    force_topologies: List[str] = None,max_workers: int = 4) -> Dict[str, Dict[str, Any]]:

    start = time.time()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    if force_topologies is None:
        force_topologies = []

    candidates = get_candidate_networks(
        bus_range=args.bus_range_test_val,
        require_switches=args.require_switches,
        max_workers=max_workers
    )
    
    if not candidates:
        raise ValueError("No suitable networks found.")

    available_forced = []
    for topology in force_topologies:
        if topology in candidates:
            available_forced.append(topology)
        else:
            logger.warning(f"Forced topology '{topology}' not found in candidates. Skipping.")

    forced_cases = []
    for i, topology in enumerate(available_forced):
        net = candidates[topology]
        dataset_tag = dataset_names[0] if dataset_names else "dataset"
        name, data = create_network_case(topology, net, dataset_tag, i, args.load_variation_range)
        if name:
            forced_cases.append((name, data, topology))
    
    logger.info(f"Created {len(forced_cases)} forced topology cases from {len(available_forced)} available topologies")

    def allocate_dataset(dataset_name: str, target_samples: int, forced_cases_for_dataset: List = None):
        """Allocate samples for a single dataset"""
        if forced_cases_for_dataset is None:
            forced_cases_for_dataset = []
        
        counts = {k: 0 for k in candidates}
        ds: Dict[str, Any] = {}
        used_topologies = set()
        
    
        for name_forced, data_forced, topology in forced_cases_for_dataset:
            ds[name_forced] = data_forced
            used_topologies.add(topology)

        remaining_candidates = {k: v for k, v in candidates.items() if k not in used_topologies}
        
        while len(ds) < target_samples:
            if not remaining_candidates:
                logger.warning(f"Ran out of unique topologies for {dataset_name}. Generated {len(ds)}/{target_samples} samples.")
                break
                
            for key, net in remaining_candidates.items():
                if len(ds) >= target_samples:
                    break
                    
                idx = counts[key]
                name, data = create_network_case(key, net, dataset_name, idx, args.load_variation_range)
                if name and name not in ds:
                    ds[name] = data
                    counts[key] += 1
                    if key in remaining_candidates:
                        del remaining_candidates[key]
                    break
        
        return ds

    # Generate datasets
    all_datasets = {}
    
    for i, (dataset_name, target_samples) in enumerate(zip(dataset_names, samples_per_dataset)):
        dataset_forced_cases = forced_cases if i == 0 else []

        adjusted_target = target_samples
        if dataset_forced_cases:
            adjusted_target = max(0, target_samples - len(dataset_forced_cases))
        
        dataset = allocate_dataset(dataset_name, adjusted_target, dataset_forced_cases)
        all_datasets[dataset_name] = dataset
        
        logger.info(f"Generated {len(dataset)} samples for {dataset_name} dataset (target: {target_samples})")

    if len(force_topologies) > 0 and len(dataset_names) > 1:
        if len(available_forced) > 0:
            logger.info(f"Forced topologies were only added to the first dataset ('{dataset_names[0]}')")
        
        unfilled_datasets = []
        for i, (dataset_name, target_samples) in enumerate(zip(dataset_names, samples_per_dataset)):
            if i > 0 and len(all_datasets[dataset_name]) < target_samples:
                unfilled_datasets.append(dataset_name)
        
        if unfilled_datasets:
            logger.warning(f"Unable to fully create datasets: {unfilled_datasets} due to topology constraints")

    # Save datasets
    now = datetime.now()
    min_bus_range = args.bus_range_test_val[0]
    max_bus_range = args.bus_range_test_val[1]
    bstr = f"{min_bus_range}-{max_bus_range}"
    
    # Create base name with dataset info
    dataset_info = "_".join([f"{name}-{count}" for name, count in zip(dataset_names, samples_per_dataset)])
    base_name = f"flexible_datasets__range-{bstr}_{dataset_info}_{now.day}{now.month}{now.year}"
    base_abs = os.path.abspath(args.data_dir)
    
    # Find sequence
    pattern = os.path.join(base_abs, base_name + '_*')
    existing = glob.glob(pattern)
    seq = 1
    if existing:
        nums = [int(os.path.basename(d).split('_')[-1]) for d in existing if d.split('_')[-1].isdigit()]
        if nums:
            seq = max(nums) + 1
    save_path = os.path.join(base_abs, f"{base_name}_{seq}")

    base_path = Path(save_path)
    os.makedirs(base_path, exist_ok=True)
    
    # Save metadata
    with open(os.path.join(base_path, 'metadata.txt'), 'w') as f:
        f.write(f"Created: {now}\n")
        f.write(f"Bus range: {args.bus_range_test_val}\n")
        f.write(f"Forced topologies: {force_topologies}\n")
        for name, samples in zip(dataset_names, samples_per_dataset):
            actual_count = len(all_datasets[name])
            f.write(f"{name}: {actual_count}/{samples}\n")
    
    # Save each dataset
    for dataset_name, dataset_data in all_datasets.items():
        save_combined_data(dataset_data, dataset_name, base_path)
    
    logger.info(f"All datasets saved at {base_path}")
    logger.info(f"Generated datasets in {time.time()-start:.1f}s")

    return all_datasets
def save_combined_data(dataset, set_name, base_dir):
    pp_dir = base_dir / set_name / "pandapower_networks"
    os.makedirs(pp_dir, exist_ok=True)
    for case_name, data in dataset.items():
        net = data["network"]
        pp_file = os.path.join(pp_dir, f"{case_name}.json")
        with open(pp_file, "w") as f:
            json.dump(pp.to_json(net), f)
    
    logger.info(f"Saved {len(dataset)} {set_name} cases individually to {base_dir}")

def create_network_case(network_id, net, case_type, case_idx, load_variation_range=(0.5, 1.51)):
    """Create a single network case using SimBench profiles when available"""
    global network_timepoints
    
    try:
        net_case = copy.deepcopy(net)
        used_profile = False
        selected_timepoint = None
        if network_id.startswith('simbench_'):
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
                    logger.debug(f"Found {len(available_timesteps)} timepoints for {network_id}")
                    
                # Get unused timepoints
                available = [tp for tp in network_timepoints[network_id]['timepoints'] 
                            if tp not in network_timepoints[network_id]['used']]
                    
                if available:
                    # Select a random unused timepoint
                    selected_timepoint = random.choice(available)
                    network_timepoints[network_id]['used'].add(selected_timepoint)
                    
                    # Apply the timepoint to the network - this is the correct way
                    net_case = apply_profile_timepoint(net_case, profiles, selected_timepoint)
                    
                    logger.info(f"Successfully applied SimBench profile timepoint {selected_timepoint} to {network_id}")
                    used_profile = True
                else:
                    logger.info("all available timepoints used applying random variation to exsisting timepoints")
                    selected_timepoint = random.choice(network_timepoints[network_id]["used"])
                    net_case = apply_profile_timepoint(net_case, profiles, selected_timepoint)
                    net_case = apply_random_load_variations(net_case, load_variation_range)

            except Exception as e:
                logger.info(f"Error accessing SimBench profiles: {e}")
                net_case = apply_random_load_variations(net_case, load_variation_range)
        else:
            net_case = apply_random_load_variations(net_case, load_variation_range)

        G_nx = top.create_nxgraph(net_case, respect_switches=False)
        components = list(nx.connected_components(G_nx))

        # pick the largest
        largest = max(components, key=len)
        logger.debug(f"[DEBUG] keeping largest component: {len(largest)} buses of {G_nx.number_of_nodes()}")
        
        # now mask your net to that
        subnet = pp.select_subnet(net_case, buses=list(largest), include_switch_buses=True, keep_everything_else =True)
        
        nx_graph_subnet = top.create_nxgraph(subnet, respect_switches=True)
        node_count = len(subnet.bus)
        logger.debug(f"[DEBUG] amount of nodes of created subnet: {node_count}")
        # Create case name - include timepoint in name if one was used
        if selected_timepoint is not None and used_profile:
            case_name = f"{network_id}_{case_type}_{case_idx}_ts{selected_timepoint}_n{node_count}"
        else:
            case_name = f"{network_id}_{case_type}_{case_idx}_n{node_count}"

        switch_count = len(subnet.switch) if hasattr(subnet, 'switch') else 0
        logger.debug(f"Added {case_type} case {case_name} with {switch_count} switches")

        return case_name, {"network": subnet, "nx_graph": nx_graph_subnet}
    except Exception as e:
        logger.info(f"Failed to create case for {network_id}: {e}")
        return None, None

def reset_timepoints():
    global network_timepoints
    network_timepoints = {}

