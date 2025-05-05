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
    """
    Attach a FileHandler to the 'network_optimizer' logger in *every*
    process (main + workers), appending to SHARED_LOG_PATH.
    """
    logger = logging.getLogger("network_optimizer")
    logger.setLevel(logging.DEBUG)
    # (avoid adding duplicates if someone re‐calls this)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(SHARED_LOG_PATH)
               for h in logger.handlers):
        fh = logging.FileHandler(SHARED_LOG_PATH, mode="a",
                                 encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = "%(asctime)s - %(levelname)s - %(processName)s - %(message)s"
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    # optional: silence propagation so we don’t double‐up on console
    logger.propagate = False

def init_logging(method):
    # 1) Grab your specific logger and configure it only—no basicConfig
    logger = logging.getLogger("network_optimizer")
    logger.setLevel(logging.INFO)
    logger.propagate = False              # don’t bubble up to root

    # 2) File handler at DEBUG
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{method.upper()}_logs.txt",
                             mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # 3) Stream handler at DEBUG (or INFO if you prefer less console spam)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

    # 4) Silence all matplotlib logging below WARNING
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
    """
    Computes a penalised MST on the *energised* buses only
    and updates net.switch['closed'] in-place.

    Parameters
    ----------
    net : pandapowerNet (will be modified in-place)
    penalty : float           # as before
    y_mask : pandas.Series / dict / None
        1 ↦ bus must be in the tree, 0 ↦ bus can be ignored.
        If None → every bus is treated as energised.
    """
    # ---------- prepare mask ------------------------------------------
    if y_mask is None:
        active_bus = pd.Series(1, index=net.bus.index)
    else:                                 # accept dict, Series, ndarray…
        active_bus = pd.Series(y_mask, index=net.bus.index).fillna(0).astype(bool)

    # ---------- collect meta ------------------------------------------
    line_df   = net.line.copy()
    switch_df = net.switch.copy()
    init_stat = switch_df['closed'].copy()
    bus_name  = net.bus['name'].to_dict()

    # map line-name → list of switch indices
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

    # each physical line becomes an edge **only if both end-buses active**
    for ln in line_df.itertuples():
        if not (active_bus[ln.from_bus] and active_bus[ln.to_bus]):
            continue                      # ignore islands / inactive buses
        w  = ln.r_ohm_per_km * ln.length_km
        if any(init_stat[s] for s in lines_with_switches.get(ln.name, [])):
            w += penalty                 # penalise currently closed switch
        G.add_edge(bus_name[ln.from_bus], bus_name[ln.to_bus],
                   weight=w, line_name=ln.name)

    # early exit: if G has <2 nodes nothing to do
    if G.number_of_nodes() < 2:
        for s in switch_df.index:          # open every switch to be safe
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

    # every line whose *either* end bus is inactive is forced open
    for s, sw in switch_df.query("et=='l'").iterrows():
        fb = net.line.at[sw.element, 'from_bus']
        tb = net.line.at[sw.element, 'to_bus']
        if not (active_bus[fb] and active_bus[tb]):
            net.switch.at[s, 'closed'] = False

    return net


def store_snapshots(
    graph_id: str,
    root_folder: Path,
    logger,
    **nets,
):
    """
    Parameters
    ----------
    graph_id      : unique id / filename stem of the graph
    root_folder   : root of the data set (Path)
    logger        : std. logger for progress messages
    **nets        : keyword pairs  phase_name = pandapowerNet
                    e.g.  original=net_orig, mst=net_mst, optimised=net_opt
    """
    for phase, net in nets.items():
        out_dir = root_folder / phase / "pandapower_networks"
        out_dir.mkdir(parents=True, exist_ok=True)

        fname = out_dir / f"{graph_id}.json"          # or f"{graph_id}_{phase}.json"
        with open(fname, "w") as fp:
            fp.write(pp.to_json(net))

        logger.info(f"[{graph_id}] snapshot '{phase}' saved → {fname}")
def _potential_graph(net):
    """
    Graph of *possible* current paths.
    Edges are added irrespective of switch status, so that we keep
    every element that *could* become energised.
    """
    G = nx.Graph()
    G.add_nodes_from(net.bus.index)

    # ------------------------------------------------------------------
    # 1) Lines ----------------------------------------------------------
    for idx, ln in net.line.iterrows():            # idx is line index
        if not ln.in_service:
            continue
        G.add_edge(ln.from_bus, ln.to_bus,
                   element=("line", idx))

    # ------------------------------------------------------------------
    # 2) Two-winding transformers --------------------------------------
    for idx, tf in net.trafo.iterrows():
        if not tf.in_service:
            continue
        G.add_edge(tf.hv_bus, tf.lv_bus,
                   element=("trafo", idx))

    # ------------------------------------------------------------------
    # 3) Three-winding transformers  (connect hv-mv and hv-lv) ---------
    for idx, tf in net.trafo3w.iterrows():
        if not tf.in_service:
            continue
        G.add_edge(tf.hv_bus, tf.mv_bus,
                   element=("trafo3w_hm", idx))
        G.add_edge(tf.hv_bus, tf.lv_bus,
                   element=("trafo3w_hl", idx))
        # (hv–mv–lv triangle is enough, the mv-lv edge is redundant)

    # ------------------------------------------------------------------
    # 4) Impedances -----------------------------------------------------
    for idx, imp in net.impedance.iterrows():
        if not imp.in_service:
            continue
        G.add_edge(imp.from_bus, imp.to_bus,
                   element=("impedance", idx))

    # ------------------------------------------------------------------
    # 5) Bus-bus and inline switches  (regardless of 'closed') ---------
    for s, sw in net.switch.iterrows():
        if sw.et == "b":        # bus-bus
            G.add_edge(sw.bus, sw.element,
                       element=("switch_b", s))
        elif sw.et == "l":      # inline → reuse line data
            ln = net.line.loc[sw.element]
            G.add_edge(ln.from_bus, ln.to_bus,
                       element=("switch_l", s))
    return G

def is_radial_and_connected(net, y_mask=None, require_single_ref=False):
    """
    Returns (is_radial, is_connected) **on the energised sub-graph**.
    A network is radial if each connected component is a tree with exactly one reference bus.
    
    y_mask : 1/0 per bus (Series / dict / ndarray).  None ⇒ all 1.
    """
    if y_mask is None:
        active_bus = pd.Series(1, index=net.bus.index)
    else:
        active_bus = pd.Series(y_mask, index=net.bus.index).fillna(0).astype(bool)

    # Build graph of *closed* lines between active buses
    G = nx.Graph()
    G.add_nodes_from(net.bus.index[active_bus])

    for _, sw in net.switch.query("et=='l' and closed").iterrows():
        ln = net.line.loc[sw.element]
        if active_bus[ln.from_bus] and active_bus[ln.to_bus]:
            G.add_edge(ln.from_bus, ln.to_bus)

    if G.number_of_nodes() == 0:
        return True, True       # vacuously radial & connected

    # Get reference buses (both ext_grid and slack generators)
    ref_buses = set(net.ext_grid.bus.tolist())
    if "slack" in net.gen.columns:
        ref_buses |= set(net.gen[net.gen.slack].bus)
    
    # Filter reference buses to only include active ones
    ref_buses = ref_buses & set(G.nodes())
    
    components = list(nx.connected_components(G))
    if not components:
        return True, True
    largest_component = max(components, key=len)
    G_largest = G.subgraph(largest_component)

    # # Now perform the radial and connected check on G_largest
    # is_connected = nx.is_connected(G_largest)
    # is_radial = True
    # for component in nx.connected_components(G_largest):
    #     comp_graph = G_largest.subgraph(component)
    #     comp_refs = ref_buses & component
    #     if len(comp_refs) != 1 or not nx.is_tree(comp_graph):
    #         is_radial = False
    #         break
    # return is_radial, is_connected
    components = list(nx.connected_components(G))
    if not components:
        return True, True

    # Connectedness is always evaluated on the largest energised component
    is_connected = nx.is_connected(G.subgraph(max(components, key=len)))

    if require_single_ref:          # original strict version
        for comp in components:
            comp_refs = ref_buses & comp
            if len(comp_refs) != 1 or not nx.is_tree(G.subgraph(comp)):
                return False, is_connected
        return True, is_connected

    # ← default: ignore how many reference buses live in each tree
    return all(nx.is_tree(G.subgraph(comp)) for comp in components), is_connected


def reduce_to_slack_component(net):
    """
    Keep **all** buses that belong to a connected component
    containing at least one reference bus (ext_grid or gen.slack == True).

    Returns (net_red, (rad_full, conn_full), (rad_red, conn_red))
    """
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
        #lines  = list(keep_lines),    # <- NEW
        include_switch_buses = True,
        include_results      = False,
    )

    # -- 4  stats for reduced graph
    G_red     =  _potential_graph(net)     
    rad_red   = nx.is_tree(G_red)
    conn_red  = nx.is_connected(G_red)

    return net_red, (rad_full, conn_full), (rad_red, conn_red)

# def process_single_graph(graph_id, net_json, folder_path,
#                          toggles, logger=None):
#     # ------------------------------------------------------------------
#     net_orig = pp.from_json_string(net_json)

#     if logger is None:
#         logger = logging.getLogger("network_optimizer")
    
#     # ------------------------------------------------------------------
#     # 1  ─ original PF
#     pp.runpp(net_orig, enforce_q_lims=False)
#     if not net_orig.converged: 
#         logger.warning(f"{graph_id}: PF on original net failed – skip")
#         return None

#     # ------------------------------------------------------------------
#     # 2  ─ reduce to LCC

#     net_lcc, (rad_full, conn_full), (rad_lcc, conn_lcc) = \
#         reduce_to_slack_component(net_orig)

#     bus_mask = net_orig.bus.index.isin(net_lcc.bus.index).astype(int)
    
#     pp.runpp(net_lcc, enforce_q_lims=False)
#     if not net_lcc.converged:
#         logger.warning(f"{graph_id}: PF on LCC failed – skip")
#         return None
#     #logger.info(f"{graph_id}: original (rad,conn)=({rad_full},{conn_full})  "
#     #            f"LCC (rad,conn)=({rad_lcc},{conn_lcc})")

#     # ------------------------------------------------------------------
#     # 3  ─ apply MST
#     net_mst = alternative_mst_reconfigure(copy.deepcopy(net_orig), penalty=1.0, y_mask=bus_mask)
#     pp.runpp(net_mst, enforce_q_lims=False)
#     if not net_mst.converged:
#         logger.warning(f"{graph_id}: PF after MST failed – skip")
#         return None

#     # ------------------------------------------------------------------
#     # 4  ─ SOCP optimisation starting from MST net
#     net_before = copy.deepcopy(net_mst)
#     optimizer = SOCP_class(net_mst, graph_id, logger=logger, toggles=toggles,  active_bus_mask=bus_mask)
#     optimizer.initialize()
#     model        = optimizer.create_model()
#     model.write(f"model_{optimizer.graph_id}.lp",
#                 io_options={"symbolic_solver_labels": True})
#     start        = time.time()
#     solver_res   = optimizer.solve()
#     opt_time     = time.time() - start
#     net_opt      = optimizer.update_network()   # returns the updated net
#     verify_solution(optimizer.model, tol=1e-6, logger=logger)
#     pp.runpp(net_opt, enforce_q_lims=False)

#     # ------------------------------------------------------------------
#     # 5  ─ statistics
#     flips_orig_lcc = count_switch_changes(net_orig, net_lcc)
#     flips_lcc_mst  = count_switch_changes(net_lcc,  net_mst)
#     flips_mst_opt  = count_switch_changes(net_before,  net_opt)

#     logger.info(f"{graph_id}: flips  orig→LCC={flips_orig_lcc}  "
#                 f"LCC→MST={flips_lcc_mst}  MST→OPT={flips_mst_opt}")
    
#     num_switches_changed = optimizer.num_switches_changed 

#     # ------------------------------------------------------------------
#     # 6  ─ store every snapshot
#     store_snapshots(
#         graph_id,
#         folder_path,
#         logger,
#         original = net_orig,
#         lcc      = net_lcc,
#         mst      = net_mst,
#         optimised= net_opt,
#     )

    # return {"graph_id": graph_id,
    #         "total_switches": net_orig.switch.shape[0],
    #         "optimization_time": opt_time,
    #         "switches_changed_mst": flips_lcc_mst,
    #         "rad_mst": is_radial_and_connected(net_before, y_mask=bus_mask),
    #         "switches_changed_opt": flips_mst_opt,
    #         "switches_changed": num_switches_changed,
    #         "rad_opt": is_radial_and_connected(net_opt, y_mask=bus_mask)}


def process_single_graph(graph_id, net_json, folder_path,
                         toggles, logger=None):
    # ------------------------------------------------------------------
    # ensure we have a logger
    if logger is None:
        logger = logging.getLogger("network_optimizer")
        logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # 1 — load & PF original
    net_orig = pp.from_json_string(net_json)
    pp.runpp(net_orig, enforce_q_lims=False)
    if not net_orig.converged:
        logger.warning(f"{graph_id}: PF on original net failed – skip")
        return None

    # ------------------------------------------------------------------
    # 2 — reduce to largest connected slack component
    net_lcc, (rad_full, conn_full), (rad_lcc, conn_lcc) = \
        reduce_to_slack_component(net_orig)
    bus_mask = net_orig.bus.index.isin(net_lcc.bus.index).astype(int)

    pp.runpp(net_lcc, enforce_q_lims=False)
    if not net_lcc.converged:
        logger.warning(f"{graph_id}: PF on LCC failed – skip")
        return None

    # ------------------------------------------------------------------
    # 3a — MST + optimize from MST
    net_mst = alternative_mst_reconfigure(copy.deepcopy(net_orig),
                                          penalty=1.0,
                                          y_mask=bus_mask)
    print("radial connected mst ", is_radial_and_connected(net_mst, y_mask=bus_mask))
    pp.runpp(net_mst, enforce_q_lims=False)
    if not net_mst.converged:
        logger.warning(f"{graph_id}: PF after MST failed – skip")
        return None

    net_before_mst_opt = copy.deepcopy(net_mst)
    optimizer_mst = SOCP_class(net_mst,
                               graph_id + "_mst",
                               toggles=toggles,
                               logger=logger,
                               active_bus_mask=bus_mask)
    optimizer_mst.initialize()
    optimizer_mst.model = optimizer_mst.create_model()
    t0 = time.time()
    from pyomo.util.infeasible import log_infeasible_constraints
    log_infeasible_constraints(optimizer_mst.model , log_expression=True, log_variables=True)  
    optimizer_mst.solve()
    mst_opt_time = time.time() - t0
    net_opt_mst = optimizer_mst.update_network()
    optimizer_mst.verify_solution( tol=1e-6, logger=logger)
    pp.runpp(net_opt_mst, enforce_q_lims=False)

    flips_lcc_to_mst      = count_switch_changes(net_lcc,      net_mst)
    flips_mst_to_mst_opt  = count_switch_changes(net_before_mst_opt, net_opt_mst)
    loss_improvement_mst = loss_improvement(net_before_mst_opt, net_opt_mst)


    # ------------------------------------------------------------------
    # 3b — PF original + optimize from original PF
    net_pf = copy.deepcopy(net_orig)
    pp.runpp(net_pf, enforce_q_lims=False)
    # We already checked orig PF, so this *should* converge
    optimizer_pf = SOCP_class(net_pf,
                              graph_id + "_origPF",
                              toggles=toggles,
                              logger=logger,
                              active_bus_mask=bus_mask)
    optimizer_pf.initialize()
    optimizer_pf.model = optimizer_pf.create_model()
    t1 = time.time()
    from pyomo.util.infeasible import log_infeasible_constraints
    log_infeasible_constraints(optimizer_pf.model, log_expression=True, log_variables=True)  
    optimizer_pf.solve()
    pf_opt_time = time.time() - t1
    net_opt_pf = optimizer_pf.update_network()
    optimizer_pf.verify_solution( tol=1e-6, logger=logger)
    pp.runpp(net_opt_pf, enforce_q_lims=False)

    flips_orig_to_pf_opt = count_switch_changes(net_orig, net_opt_pf)
    loss_improvement_orig = loss_improvement(net_orig, net_opt_pf)
    # ------------------------------------------------------------------
    # 4 — compare the two final states
    flips_pfopt_vs_mstopt = count_switch_changes(net_opt_pf, net_opt_mst)

    logger.info(
        f"{graph_id}: MST→OPT flips={flips_mst_to_mst_opt}  "
        f"origPF→OPT flips={flips_orig_to_pf_opt}  "
        f"state_diff(OPT_orig,OPT_mst)={flips_pfopt_vs_mstopt}"
    )

    # ------------------------------------------------------------------
    # 5 — store snapshots
    store_snapshots(
        graph_id,
        folder_path,
        logger,
        original    = net_orig,
        lcc         = net_lcc,
        mst         = net_mst,
        optimised_mst = net_opt_mst,
        optimised_origPF = net_opt_pf,
    )

    # ------------------------------------------------------------------
    # 6 — return metrics
    return {
        "graph_id": graph_id,
        "total_switches": net_orig.switch.shape[0],
        "opt_time_mst": mst_opt_time,
        "opt_time_origPF": pf_opt_time,
        "switches_changed_lcc_to_mst": flips_lcc_to_mst,
        "switches_changed_mst_to_opt": flips_mst_to_mst_opt,
        "switches_changed_orig_to_opt": flips_orig_to_pf_opt,
        "state_diff_between_opts": flips_pfopt_vs_mstopt,
        "rad_mst_opt": is_radial_and_connected(net_opt_mst, y_mask=bus_mask),
        "rad_origPF_opt": is_radial_and_connected(net_opt_pf, y_mask=bus_mask),
        "loss_improvement_mst": loss_improvement_mst["loss_improvement"],
        "loss_improvement_orig": loss_improvement_orig["loss_improvement"],
    }


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

def apply_optimization(folder_path, method="SOCP", toggles=None, debug=False):
    folder_path = Path(folder_path)
    init_worker_logging()

    _, pp_networks, features = load_graph_data_old(folder_path)

    items = [(gid, net) for gid, net in pp_networks.items()]

    metrics = []

    # --- Parallel execution ---
    with ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        initializer=init_worker_logging
    ) as executor:
        futures = {
            executor.submit(
                process_single_graph,
                gid, net_json, folder_path,
                toggles, # + pass logger if you want, but not needed now
            ): gid
            for gid, net_json in items
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            res = fut.result()
            if res:
                metrics.append(res)

    # --- Sequential execution ---
    for gid, net_json in items:
        res = process_single_graph(gid, net_json, folder_path , toggles)
        if res: metrics.append(res)

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
        print(f"Average optimization time: {metrics_df['optimization_time'].mean():.4f} seconds")
        print(f"Total switches changed: {metrics_df['switches_changed'].sum()}")
    print("="*50)


    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    if 'switches_changed' in metrics_df:
        axes[0].hist(metrics_df['switches_changed'], bins=10, alpha=0.7)
        axes[0].set_title('Switches Changed per Graph')
        axes[0].set_xlabel('Switches Changed')
        axes[0].set_ylabel('Frequency')
    else:
        axes[0].text(0.5, 0.5, 'No switch data', ha='center')
        axes[0].set_title('Switches Changed per Graph')

    if 'optimization_time' in metrics_df:
        axes[1].hist(metrics_df['optimization_time'], bins=10, alpha=0.7)
        axes[1].set_title('Optimization Time per Graph (s)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency')
    else:
        axes[1].text(0.5, 0.5, 'No time data', ha='center')
        axes[1].set_title('Optimization Time per Graph')

    plt.tight_layout()
    plt.show()
   

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
                print(f"{constr.name} (index: {constr.index() if hasattr(constr, 'index') else ''}) violation: {violation:.4e}, body value: {body_val:.4e}, bounds: ({lower}, {upper})")
        except Exception as e:
            print(f"Could not evaluate constraint {constr.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ground truth data for power networks using optimization')
    parser.add_argument('--folder_path',
                        default = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test_val_real__range-30-150_nTest-10_nVal-10_2732025_3/test/original",
                        type=str, help='Dataset folder path')
    parser.add_argument('--set', type=str, choices=['test', 'validation', 'train', '', 'all'], default='', help='Dataset set to process; leave empty for no subfolder')
    parser.add_argument('--method', type=str, choices=['SOCP', 'MILP'], default='SOCP', help='Choose optimization method: SOCP or MILP')
    parser.add_argument('--debug', type=bool, default=True, help='Print debug information')

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
        apply_optimization(Path(args.folder_path) / args.set, method=args.method, toggles=SOCP_toggles, debug=args.debug)
    elif args.set == "all": 
        for set_name in Path(args.folder_path).iterdir():
            if set_name.is_dir():
                print("\nProcessing set:", set_name)
                apply_optimization(Path(args.folder_path) / set_name, method=args.method, toggles=SOCP_toggles, debug=args.debug)
    else:
        apply_optimization(args.folder_path, method=args.method, toggles=SOCP_toggles, debug=args.debug)

    print("\nGround truth generation complete!!!!")
