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
import traceback
import random
import copy 

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pandapower.powerflow import LoadflowNotConverged

# Add necessary source paths
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
load_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_search"))
if src_path not in sys.path:
    sys.path.append(src_path)
if load_data_path not in sys.path:
    sys.path.append(load_data_path)


from SOCP_class_dnr import SOCP_class
from load_data import load_graph_data_old
from electrify_subgraph import extract_node_features, extract_edge_features

SHARED_LOG_PATH = Path(__file__).parent / "network_optimizer_all.log"

def init_worker_logging():
    logger = logging.getLogger("network_optimizer")
    logger.setLevel(logging.DEBUG)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(SHARED_LOG_PATH)
               for h in logger.handlers):
        fh = logging.FileHandler(SHARED_LOG_PATH, mode="a",
                                 encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = "%(asctime)s - %(levelname)s - %(processName)s - %(message)s"
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    logger.propagate = False

def init_logging(method):
    logger = logging.getLogger("network_optimizer")
    logger.setLevel(logging.INFO)
    logger.propagate = False              

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{method.upper()}_logs.txt",
                             mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)
    ml = logging.getLogger("matplotlib")
    ml.setLevel(logging.WARNING)
    mf = logging.getLogger("matplotlib.font_manager")
    mf.setLevel(logging.WARNING)

    logging.getLogger("gurobipy").setLevel(logging.WARNING)

    logger.debug(f"Logging initialized. All logs to {log_dir}/{method.upper()}_logs.txt")
    return logger

def get_n_workers():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

def alternative_mst_reconfigure(net, penalty=1.0, y_mask=None):

    # ---------- prepare mask ------------------------------------------
    if y_mask is None:
        active_bus = pd.Series(1, index=net.bus.index)
    else:                                
        active_bus = pd.Series(y_mask, index=net.bus.index).fillna(0).astype(bool)

    # ---------- collect meta ------------------------------------------
    line_df   = net.line.copy()
    switch_df = net.switch.copy()
    init_stat = switch_df['closed'].copy()
    bus_name  = net.bus['name'].to_dict()

    lines_with_switches = {}
    for s, sw in switch_df.query("et=='l'").iterrows():
        ln_idx = sw.element
        ln_name = net.line.at[ln_idx, 'name']
        lines_with_switches.setdefault(ln_name, []).append(s)

    # ---------- build candidate graph -------------------------------
    G = nx.Graph()
    for b in net.bus.itertuples():
        if active_bus[b.Index]:
            G.add_node(b.name)

    for ln in line_df.itertuples():
        if not (active_bus[ln.from_bus] and active_bus[ln.to_bus]):
            continue                     
        w  = ln.r_ohm_per_km * ln.length_km
        if any(init_stat[s] for s in lines_with_switches.get(ln.name, [])):
            w += penalty                
        G.add_edge(bus_name[ln.from_bus], bus_name[ln.to_bus],
                   weight=w, line_name=ln.name)
    if G.number_of_nodes() < 2:
        for s in switch_df.index:          
            net.switch.at[s, 'closed'] = False
        return net

    # ---------- MST & switch update ----------------------------------
    mst        = nx.minimum_spanning_tree(G)
    mst_edges  = {frozenset(e) for e in mst.edges()}

    for ln_name, sw_list in lines_with_switches.items():
        ln_row = line_df[line_df.name == ln_name].iloc[0]
        fb = bus_name[ln_row.from_bus]
        tb = bus_name[ln_row.to_bus]
        new_state = frozenset((fb, tb)) in mst_edges
        for s in sw_list:
            net.switch.at[s, 'closed'] = new_state

    for s, sw in switch_df.query("et=='l'").iterrows():
        fb = net.line.at[sw.element, 'from_bus']
        tb = net.line.at[sw.element, 'to_bus']
        if not (active_bus[fb] and active_bus[tb]):
            net.switch.at[s, 'closed'] = False

    return net


def store_snapshots(graph_id: str, root_folder: Path, logger, **nets,):
    for phase, net in nets.items():
        out_dir = root_folder / phase / "pandapower_networks"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{graph_id}.json"         
        with open(fname, "w") as fp:
            fp.write(pp.to_json(net))

        logger.info(f"[{graph_id}] snapshot '{phase}' saved → {fname}")
def _potential_graph(net):
    G = nx.Graph()
    G.add_nodes_from(net.bus.index)

    # 1) Lines ----------------------------------------------------------
    for idx, ln in net.line.iterrows():            
        if not ln.in_service:
            continue
        G.add_edge(ln.from_bus, ln.to_bus,
                   element=("line", idx))

    # 2) transformers --------------------------------------
    for idx, tf in net.trafo.iterrows():
        if not tf.in_service:
            continue
        G.add_edge(tf.hv_bus, tf.lv_bus,
                   element=("trafo", idx))

    for idx, tf in net.trafo3w.iterrows():
        if not tf.in_service:
            continue
        G.add_edge(tf.hv_bus, tf.mv_bus,
                   element=("trafo3w_hm", idx))
        G.add_edge(tf.hv_bus, tf.lv_bus,
                   element=("trafo3w_hl", idx))


    # 3) Impedances -----------------------------------------------------
    for idx, imp in net.impedance.iterrows():
        if not imp.in_service:
            continue
        G.add_edge(imp.from_bus, imp.to_bus,
                   element=("impedance", idx))


    # 4) Bus-bus and inline switches  
    for s, sw in net.switch.iterrows():
        if sw.et == "b":        
            G.add_edge(sw.bus, sw.element,
                       element=("switch_b", s))
        elif sw.et == "l":      
            ln = net.line.loc[sw.element]
            G.add_edge(ln.from_bus, ln.to_bus,
                       element=("switch_l", s))
    return G

def is_radial_and_connected(net, y_mask=None, require_single_ref=False):
    if y_mask is None:
        active_bus = pd.Series(1, index=net.bus.index)
    else:
        active_bus = pd.Series(y_mask, index=net.bus.index).fillna(0).astype(bool)

    G = nx.Graph()
    G.add_nodes_from(net.bus.index[active_bus])

    for _, sw in net.switch.query("et=='l' and closed").iterrows():
        ln = net.line.loc[sw.element]
        if active_bus[ln.from_bus] and active_bus[ln.to_bus]:
            G.add_edge(ln.from_bus, ln.to_bus)

    if G.number_of_nodes() == 0:
        return True, True       

    ref_buses = set(net.ext_grid.bus.tolist())
    if "slack" in net.gen.columns:
        ref_buses |= set(net.gen[net.gen.slack].bus)

    ref_buses = ref_buses & set(G.nodes())
    
    components = list(nx.connected_components(G))
    if not components:
        return True, True

    components = list(nx.connected_components(G))
    if not components:
        return True, True

    is_connected = nx.is_connected(G.subgraph(max(components, key=len)))

    if require_single_ref:          
        for comp in components:
            comp_refs = ref_buses & comp
            if len(comp_refs) != 1 or not nx.is_tree(G.subgraph(comp)):
                return False, is_connected
        return True, is_connected
    return all(nx.is_tree(G.subgraph(comp)) for comp in components), is_connected


def reduce_to_slack_component(net):
    # -- 0  identify reference buses
    slack_buses = slack_buses = set(net.ext_grid.bus) | \
                  set(net.gen[getattr(net.gen, "slack", False)].bus)
    if "slack" in net.gen.columns:
        slack_buses |= set(net.gen[net.gen.slack].bus.tolist())
    if not slack_buses:
        raise ValueError("No ext_grid or slack generator found in the net")

    # -- 1  full graph & stats
    G_full       = _potential_graph(net)     
    rad_full     = nx.is_tree(G_full)
    conn_full    = nx.is_connected(G_full)

    # -- 2  collect every component that touches a slack bus
    keep_nodes = set().union(
        *[comp for comp in nx.connected_components(G_full) if comp & slack_buses]
    )
    keep_lines = net.line.index[
        net.line.from_bus.isin(keep_nodes) |
        net.line.to_bus.isin(keep_nodes)
    ]

    # -- 3  build reduced net that still contains *all* slacks
    net_red = pp.select_subnet(
        net,
        buses  = list(keep_nodes),
        include_switch_buses = True,
        include_results      = False,
    )

    # -- 4  stats for reduced graph
    G_red     =  _potential_graph(net)     
    rad_red   = nx.is_tree(G_red)
    conn_red  = nx.is_connected(G_red)

    return net_red, (rad_full, conn_full), (rad_red, conn_red)


def process_single_graph(graph_id, net_json, folder_path,
                         toggles, logger=None, debug=False):
    from pyomo.util.infeasible import log_infeasible_constraints
    # ------------------------------------------------------------------
    # intit logger
    if logger is None:
        logger = logging.getLogger("network_optimizer")
        logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # 1 — load & PF original
    net_orig = pp.from_json_string(net_json)
    try: 
        pp.runpp(net_orig, enforce_q_lims=False)
        if not net_orig.converged:
            logger.warning(f"{graph_id}: PF on original net failed – skip")
            return None
    except LoadflowNotConverged:
        logger.warning(f"{graph_id}: PF on original net failed – skip")
        return None
    

    # ------------------------------------------------------------------
    # 2 — reduce to largest connected slack component 
    # simbench/pandapower have unconnected hv islands
    net_lcc, _, _ = reduce_to_slack_component(net_orig)
    bus_mask = net_orig.bus.index.isin(net_lcc.bus.index).astype(int)
    try:
        pp.runpp(net_lcc, enforce_q_lims=False)
    except LoadflowNotConverged:
        logger.warning(f"{graph_id}: PF on LCC did not converge – skip")
        return None

    # ------------------------------------------------------------------
    # 3a — MST + optimize from MST
    net_mst = alternative_mst_reconfigure(copy.deepcopy(net_orig), penalty=1.0,y_mask=bus_mask)
    print("radial connected mst ", is_radial_and_connected(net_mst, y_mask=bus_mask))
    try:
        pp.runpp(net_mst, enforce_q_lims=False)
    except LoadflowNotConverged:
        logger.warning(f"{graph_id}: PF after MST did not converge – skip")
        return None

    net_before_mst_opt = copy.deepcopy(net_mst)
    optimizer_mst = SOCP_class(net_mst, graph_id + "_mst",  toggles=toggles,
                               logger=logger, active_bus_mask=bus_mask)
    optimizer_mst.initialize()
    optimizer_mst.model = optimizer_mst.create_model()
     
    t0 = time.time()
    optimizer_mst.solve()
    mst_opt_time = time.time() - t0
    if debug:
        optimizer_mst.model.write(f"{graph_id}_mst.lp",
                              io_options={"symbolic_solver_labels": True})
        log_infeasible_constraints(optimizer_mst.model , log_expression=True, log_variables=True)
    net_opt_mst = optimizer_mst.update_network()
    optimizer_mst.verify_solution( tol=1e-6, logger=logger)
    try:
        pp.runpp(net_opt_mst, enforce_q_lims=False)
    except LoadflowNotConverged:
        logger.warning(f"{net_opt_mst}: PF after optimization of MST did not converge – skip")
        return None

    #flips_orig_lcc = count_switch_changes(net_orig, net_lcc)
    flips_lcc_mst = count_switch_changes(net_lcc, net_mst)
    flips_mst_opt = count_switch_changes(net_before_mst_opt, net_opt_mst)
    num_switches_changed = optimizer_mst.num_switches_changed
    rad_mst = is_radial_and_connected(net_before_mst_opt, y_mask=bus_mask)
    rad_opt = is_radial_and_connected(net_opt_mst, y_mask=bus_mask)
    loss_improvement_mst = loss_improvement(net_before_mst_opt, net_opt_mst)
    metrics = {
        "graph_id": graph_id,
        "total_switches": net_orig.switch.shape[0],
        "opt_time_mst": mst_opt_time,
        "switches_changed_lcc_to_mst": flips_lcc_mst,
        "switches_changed_mst_to_opt": flips_mst_opt,
        #"switches_changed": num_switches_changed,
        "loss_improvement_mst": loss_improvement_mst["loss_improvement"],
        "rad_mst": rad_mst,
        "rad_mst_opt": rad_opt,
    }
    snapshots = {
        "original": net_orig,
        "lcc": net_lcc,
        "mst": net_mst,
        "optimised_mst": net_opt_mst,
    }

    # ------------------------------------------------------------------
    # 3b — PF original + optimize from original PF
    if debug:
        net_pf = copy.deepcopy(net_orig)
        try:
            pp.runpp(net_pf, enforce_q_lims=False)
        except LoadflowNotConverged:
            logger.warning(f"{graph_id}: PF on original net failed – skip")
            return None
        # We already checked orig PF, so this *should* converge
        optimizer_pf = SOCP_class(net_pf,
                                graph_id + "_origPF",
                                toggles=toggles,
                                logger=logger,
                                active_bus_mask=bus_mask)
        optimizer_pf.initialize()
        optimizer_pf.model = optimizer_pf.create_model()
        optimizer_pf.model.write(f"{graph_id}_origPF.lp",
                                  io_options={"symbolic_solver_labels": True})
        from pyomo.util.infeasible import log_infeasible_constraints
        t1 = time.time()
        optimizer_pf.solve()
        pf_opt_time = time.time() - t1
        log_infeasible_constraints(optimizer_pf.model, log_expression=True, log_variables=True)  
        net_opt_pf = optimizer_pf.update_network()
        optimizer_pf.verify_solution( tol=1e-6, logger=logger)
        try:
            pp.runpp(net_opt_pf, enforce_q_lims=False)
        except LoadflowNotConverged:
            logger.warning(f"{graph_id}: PF after orig opt did not converge – skip")
            return None
        flips_pf_to_opt = count_switch_changes(net_pf, net_opt_pf)
        loss_improvement_pf = loss_improvement(net_pf, net_opt_pf)
        num_switches_changed = optimizer_mst.num_switches_changed
        flips_pfopt_vs_mstopt = count_switch_changes(net_opt_mst, net_opt_pf)

        metrics.update({
            "opt_time_origPF": pf_opt_time,
            "switches_changed_orig_to_opt": flips_pf_to_opt,
            "state_diff_between_opts": flips_pfopt_vs_mstopt,
            "loss_improvement_orig": loss_improvement_pf["loss_improvement"],
            "rad_origPF_opt": is_radial_and_connected(net_opt_pf, y_mask=bus_mask),
        })
        snapshots["optimised_origPF"] = net_opt_pf
        logger.info(
            f"{graph_id}: MST→OPT={flips_pf_to_opt} flips; origPF→OPT={flips_mst_opt} flips; diff={flips_pfopt_vs_mstopt}"
        )
    else:

        logger.info(f"{graph_id}: MST only, MST→OPT={flips_mst_opt} flips")

    store_snapshots( graph_id,folder_path, logger,**snapshots)

    return metrics

# {
#         "graph_id": graph_id,
#         "total_switches": net_orig.switch.shape[0],
#         "opt_time_mst": mst_opt_time,
#         "opt_time_origPF": pf_opt_time,
#         "switches_changed_lcc_to_mst": flips_lcc_to_mst,
#         "switches_changed_mst_to_opt": flips_mst_to_mst_opt,
#         "switches_changed_orig_to_opt": flips_orig_to_pf_opt,
#         "state_diff_between_opts": flips_pfopt_vs_mstopt,
#         "rad_mst_opt": is_radial_and_connected(net_opt_mst, y_mask=bus_mask),
#         "rad_origPF_opt": is_radial_and_connected(net_opt_pf, y_mask=bus_mask),
#         "loss_improvement_mst": loss_improvement_mst["loss_improvement"],
#         "loss_improvement_orig": loss_improvement_orig["loss_improvement"],
#     }


def count_switch_changes(net_a, net_b):
    """
    Count how many line‐switch statuses differ between two pandapower nets.
    """
    sa = net_a.switch[(net_a.switch.et=='l')]['closed']
    sb = net_b.switch[(net_b.switch.et=='l')]['closed']
    common = sa.index.intersection(sb.index)
    return int((sa.loc[common] != sb.loc[common]).sum())

def loss_improvement(net_before, net_after, include_trafos=True):
    """
    Return active-power loss reduction in percent (and the absolute numbers).
    """

    # run power flow once for each network
    pp.runpp(net_before, enforce_q_lims=False)
    pp.runpp(net_after,  enforce_q_lims=False)
    def active_loss(net):
        loss = net.res_line["pl_mw"].abs().sum()
        if include_trafos and "res_trafo" in net:
            loss += net.res_trafo["pl_mw"].abs().sum()
        return float(loss)

    l0 = active_loss(net_before)
    l1 = active_loss(net_after)

    return {
        "loss_before":       l0,
        "loss_after":        l1,
        "loss_improvement":  100.0 * (l0 - l1) / l0 if l0 > 0 else np.nan
    }

def store_optimized_models(
    optimizer,
    net_original,
    net_post_mst,
    net_optimized,
    graph_id,
    logger,
    opt_results,
    root_folder=Path("dataset_folder"),
    voltage_lower=0.9,
    voltage_upper=1.10,
    bus_map=None
    ):

    # Then save each snapshot as before
    snapshots = [
        ("original",  net_original, "_original"),
        ("post_MST",  net_post_mst, "_post_mst"),
        ("optimized", net_optimized, "_optimized"),
    ]
    for phase_name, snapshot, suffix in snapshots:
        pp_dir = root_folder/phase_name/"pandapower_networks"
        pp_dir.mkdir(parents=True, exist_ok=True)
        with open(pp_dir/f"{graph_id}{suffix}.json","w") as f:
            f.write(pp.to_json(snapshot))
        logger.info(f"Saved PP network ({phase_name})")

def apply_optimization(folder_path, method="SOCP", toggles=None, debug=False, serialize=False):
    folder_path = Path(folder_path)
    init_worker_logging()

    _, pp_networks, features = load_graph_data_old(folder_path)

    items = [(gid, net) for gid, net in pp_networks.items()]

    metrics = []

    if serialize:
            #--- Sequential execution ---
        for gid, net_json in items:
            res = process_single_graph(gid, net_json, folder_path , toggles, debug = debug)
            if res: metrics.append(res)
    else: # default
        # --- Parallel execution ---
        with ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            initializer=init_worker_logging
        ) as executor:
            futures = {
                executor.submit(
                    process_single_graph,
                    gid, net_json, folder_path,
                    toggles=toggles, debug=debug
                ): gid
                for gid, net_json in items
            }
            for fut in tqdm(as_completed(futures), total=len(futures)):
                gid = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    logging.getLogger("network_optimizer").error(
                        f"{gid}: failed in worker – skipping", exc_info=True
                    )
                    continue
                if res:
                    metrics.append(res)


    df = pd.DataFrame(metrics)
    df.to_csv(folder_path / "optimization_metrics.csv", index=False)
    print(df)

    # Save optimization metrics.
    metrics_df = pd.DataFrame(metrics)
    metrics_csv = folder_path / "optimization_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n✓ Saved optimization metrics to {metrics_csv}")

    # Summary printing.
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Total graphs processed: {len(pp_networks)}")
    print(f"Successful optimizations: {len(metrics)}")


    if not metrics_df.empty:
        # print basic stats for each numeric column
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # choose sum for counts, mean for times/losses
            if "time" in col or "loss" in col:
                stat = metrics_df[col].mean()
                print(f"Average {col}: {stat:.4f}")
            else:
                stat = metrics_df[col].sum()
                print(f"Total {col}: {stat}")
    print("="*50)

    # --- dynamic histograms ---
    # pick numeric columns again
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        n = len(numeric_cols)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
        axes = np.atleast_1d(axes).flatten()

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            ax.hist(metrics_df[col].dropna(), bins=10)
            ax.set_title(col.replace('_',' ').title())
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

        # turn off any unused axes
        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ground truth data for power networks using optimization')
    parser.add_argument('--folder_path',
                        default = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_val_real__range-30-150_nTest-10_nVal-10_2732025_3/test/original",
                        #default = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\28042025_range-130-100_1000_2\original",
                        #default = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_val_real__range-30-230_nTest-1000_nVal-1000_2842025_1\test_val_real__range-30-230_nTest-1000_nVal-1000_2842025\validation\original",
                        type=str, help='Dataset folder path')
    parser.add_argument('--set', type=str, choices=['test', 'validation', 'train', '', 'all'], default='', help='Dataset set to process; leave empty for no subfolder')
    parser.add_argument('--method', type=str, choices=['SOCP', 'MILP'], default='SOCP', help='Choose optimization method: SOCP or MILP')
    parser.add_argument('--debug', type=bool, default=True, help='Print debug information')
    parser.add_argument('--serialize', type=bool, default=True, help='Serialize the optimization results usefull for debugging')
    # SOCP toggles
    parser.add_argument('--include_voltage_drop_constraint', type=bool, default=True, help="Include voltage drop constraint SOCP")
    parser.add_argument('--include_voltage_bounds_constraint', type=bool, default=True, help="Include voltage bounds constraint SOCP")
    parser.add_argument('--include_power_balance_constraint', type=bool, default=True, help="Include power balance constraint SOCP")
    parser.add_argument('--include_radiality_constraints', type=bool, default=True, help="Include radiality constraints SOCP")
    parser.add_argument('--use_spanning_tree_radiality', type=bool, default=False, help="Use spanning tree radiality SOCP")
    parser.add_argument('--include_switch_penalty', type=bool, default=True, help="Include switch penalty in objective SOCP")
    
    parser.add_argument("--write_files", action="store_true", help="If set, write out LP/MPS model files; otherwise skip for speed")
    args = parser.parse_args()

    
    SOCP_toggles = { 
        "include_voltage_drop_constraint": args.include_voltage_drop_constraint, 
        "include_voltage_bounds_constraint": args.include_voltage_bounds_constraint,   
        "include_power_balance_constraint": args.include_power_balance_constraint,  
        "include_radiality_constraints": args.include_radiality_constraints,
        "use_spanning_tree_radiality": args.use_spanning_tree_radiality,  
        "include_switch_penalty": args.include_switch_penalty,
    }
    
    print("Toggles for optimization:")
    print(SOCP_toggles)
    if args.set:
        apply_optimization(Path(args.folder_path) / args.set, method=args.method, toggles=SOCP_toggles, debug=args.debug, serialize=args.serialize)
    elif args.set == "all": 
        for set_name in Path(args.folder_path).iterdir():
            if set_name.is_dir():
                print("\nProcessing set:", set_name)
                apply_optimization(Path(args.folder_path) / set_name, method=args.method, toggles=SOCP_toggles, debug=args.debug, serialize=args.serialize)
    else:
        apply_optimization(args.folder_path, method=args.method, toggles=SOCP_toggles, debug=args.debug, serialize=args.serialize)

    print("\nGround truth generation complete!!!!")
