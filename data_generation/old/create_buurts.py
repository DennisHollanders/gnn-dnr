import pickle
import ast
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# --- Helper: find project root dynamically ---
def find_project_root(marker_files=(".git", "pyproject.toml")) -> Path:
    current = Path(__file__).resolve()
    for p in (current, *current.parents):
        if any((p / m).exists() for m in marker_files):
            return p
    return current.parent

# --- Configuration ---
BASE             = find_project_root()
DATA_DIR         = BASE  / "data"
#DATA_DIR         = BASE / "gnn-dnr" / "data"
STORE_DIR        = DATA_DIR / "cbs_buurts"
STORE_DIR.mkdir(parents=True, exist_ok=True)

SUBGRAPH_FILE    = DATA_DIR / "filtered_subgraphs.pkl"
BUURT_LOOKUP     = DATA_DIR / "buurt_to_postcodes.csv"
CBS_PC6_FILE     = DATA_DIR / "cbs_pc6_2023.gpkg"
CONS_DATA_FILE   = DATA_DIR / "aggregated_kleinverbruik_with_opwek.csv"

STORE_MODE       = "node"      # or "node"
NODE_CLUSTER_SZ  = 10
MAX_PER_CLUSTER  = 100

# --- Phase A: load & centroid + buurt mapping ---
with open(SUBGRAPH_FILE, "rb") as f:
    all_subgraphs = pickle.load(f)

clean_subgraphs = []
centroids = []
for sg in tqdm(all_subgraphs, desc="Phase A: centroids"):
    G = sg.copy()
    for _, d in G.nodes(data=True):
        d.pop("some_large_feature_array", None)
    clean_subgraphs.append(G)

    pos = [d["position"] for _, d in G.nodes(data=True)
           if isinstance(d.get("position"), (list, tuple)) and len(d["position"]) == 2]
    if pos:
        centroids.append(Point(np.mean(pos, axis=0)))
    else:
        geoms = [d.get("geometry") for _, d in G.nodes(data=True) if d.get("geometry")]
        xs = [g.x for g in geoms]; ys = [g.y for g in geoms]
        centroids.append(Point(np.mean(xs), np.mean(ys)))

centroid_gdf = gpd.GeoDataFrame(
    {"subgraph_index": range(len(clean_subgraphs))},
    geometry=centroids,
    crs="EPSG:28992"
)

cbs_pc6 = gpd.read_file(CBS_PC6_FILE)[["postcode6", "geometry"]]
if cbs_pc6.crs != centroid_gdf.crs:
    cbs_pc6 = cbs_pc6.to_crs(centroid_gdf.crs)

centroid_pc6 = gpd.sjoin_nearest(
    centroid_gdf, cbs_pc6,
    how="left", distance_col="distance"
)[["subgraph_index", "postcode6"]]

buurt = pd.read_csv(BUURT_LOOKUP)
buurt["postcode6"] = (
    buurt["postcode6"]
    .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    .apply(lambda x: x if isinstance(x, (list, tuple)) else [x])
)
expanded = (
    buurt.explode("postcode6")
    .assign(postcode6=lambda df: df["postcode6"].astype(str))
    .rename(columns={"buurtcode": "buurt"})
    [["buurt", "postcode6"]]
)

mapping = centroid_pc6.merge(expanded, on="postcode6", how="left")
mapping.to_csv(STORE_DIR / "subgraph_to_buurt.csv", index=False)

node_counts = defaultdict(list)
for _, row in mapping.dropna(subset=["buurt"]).iterrows():
    idx = int(row["subgraph_index"])
    node_counts[row["buurt"]].append(len(clean_subgraphs[idx]))
with open(STORE_DIR / "subgraphs_node_counts.json", "w", encoding="utf-8") as jf:
    json.dump(node_counts, jf, indent=2, ensure_ascii=False)

print("[Phase A] mapping + JSON written.")

# --- Phase B: save pickles by buurt or by node-count clusters ---
if STORE_MODE == "buurt":
    mapping = pd.read_csv(STORE_DIR / "subgraph_to_buurt.csv")
    grouped = defaultdict(list)
    for _, row in mapping.dropna(subset=["buurt"]).iterrows():
        idx = int(row["subgraph_index"])
        grouped[row["buurt"]].append(clean_subgraphs[idx])

    cons_df = pd.read_csv(CONS_DATA_FILE)
    metrics_json = {}
    saved = 0

    for buurt_code, subgs in tqdm(grouped.items(), desc="Phase B (buurt)"):
        total_nodes = sum(len(sg.nodes) for sg in subgs)
        neigh = cons_df[cons_df["CBS Buurtcode"] == buurt_code]
        if neigh.empty:
            continue

        sja  = neigh["kleinverbruik_SJA_GEMIDDELD"].iloc[0]
        met  = neigh["opwek_klein_AANTAL_AANSLUITINGEN_MET_OPWEKINSTALLATIE"].iloc[0]
        tot  = neigh["opwek_klein_AANTAL_AANSLUITINGEN_IN_CBS_BUURT"].iloc[0]
        aans = neigh["kleinverbruik_AANSLUITINGEN_AANTAL"].iloc[0]
        pct  = (met + 1) / (tot + 1) if tot >= 0 else 0

        groot_cols = [
            "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_WIND_IN_AANLEG",
            "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_WIND_IN_BEDRIJF",
            "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_ZON_IN_AANLEG",
            "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_ZON_IN_BEDRIJF"
        ]
        groot_sum = neigh[groot_cols].sum(axis=1).fillna(0).iloc[0]

        hours, kw_mw = 8760, 1000
        total_kwh = sja * aans
        avg_cons   = (total_kwh / hours) / kw_mw
        avg_gen    = (groot_sum / hours) / kw_mw
        avg_per_node = avg_cons / (total_nodes + 1e-6)

        buurt_data = {
            'subgraphs': subgs,
            'info': {
                'total_nodes': total_nodes,
                'nodes_per_subgraph': [len(sg.nodes) for sg in subgs],
                'percentage_gen_small': pct,
                'avg_power_mw_generation_groot': avg_gen,
                'avg_power_mw_consumption_small_per_node': avg_per_node,
                'original_opwek_groot_kwh_yearly': groot_sum,
                'original_ja_kwh_buurt_small': total_kwh
            }
        }

        fname = STORE_DIR / f"{buurt_code}_n-{len(subgs)}.pkl"
        with open(fname, 'wb') as out:
            pickle.dump(buurt_data, out, protocol=pickle.HIGHEST_PROTOCOL)

        metrics_json[buurt_code] = [len(sg.nodes) for sg in subgs]
        saved += 1

    with open(STORE_DIR / "subgraphs_node_counts_updated.json", 'w', encoding='utf-8') as jf:
        json.dump(metrics_json, jf, indent=2, ensure_ascii=False)

    print(f"[Phase B] buurt mode done. Saved {saved} files.")

elif STORE_MODE == "node":
    def chunker(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    buckets = defaultdict(list)
    for idx, G in enumerate(clean_subgraphs):
        n = G.number_of_nodes()
        lower = (n // NODE_CLUSTER_SZ) * NODE_CLUSTER_SZ
        upper = lower + NODE_CLUSTER_SZ
        label = f"n{str(lower).zfill(2)}_{str(upper).zfill(2)}"
        buckets[label].append(idx)

    for label, idx_list in buckets.items():
        for part, chunk in enumerate(chunker(idx_list, MAX_PER_CLUSTER), start=1):
            subgs = [clean_subgraphs[i] for i in chunk]
            fname = STORE_DIR / f"subgraph-{label}-{part}-n{len(subgs)}.pkl"
            with open(fname, 'wb') as out:
                pickle.dump(subgs, out, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Phase B] node mode done. {len(buckets)} buckets written.")

else:
    raise ValueError(f"Unknown STORE_MODE={STORE_MODE!r}")




"""
import pickle
import ast
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# --- Helper: find project root dynamically ---
def find_project_root(marker_files=(".git", "pyproject.toml")) -> Path:
    current = Path(__file__).resolve()
    for p in (current, *current.parents):
        if any((p / m).exists() for m in marker_files):
            return p
    return current.parent

# --- Configuration ---
BASE             = find_project_root()
DATA_DIR         = BASE  / "data"
#DATA_DIR         = BASE / "gnn-dnr" / "data"
STORE_DIR        = DATA_DIR / "cbs_buurts"
STORE_DIR.mkdir(parents=True, exist_ok=True)

SUBGRAPH_FILE    = DATA_DIR / "filtered_complete_subgraphs_final.pkl"
BUURT_LOOKUP     = DATA_DIR / "buurt_to_postcodes.csv"
CBS_PC6_FILE     = DATA_DIR / "cbs_pc6_2023.gpkg"
CONS_DATA_FILE   = DATA_DIR / "aggregated_kleinverbruik_with_opwek.csv"

STORE_MODE       = "buurt"      # or "node"
NODE_CLUSTER_SZ  = 10
MAX_PER_CLUSTER  = 100

# --- Phase A: load & centroid + buurt mapping ---
with open(SUBGRAPH_FILE, "rb") as f:
    all_subgraphs = pickle.load(f)

clean_subgraphs = []
centroids = []
for sg in tqdm(all_subgraphs, desc="Phase A: centroids"):
    G = sg.copy()
    for _, d in G.nodes(data=True):
        d.pop("some_large_feature_array", None)
    clean_subgraphs.append(G)

    pos = [d["position"] for _, d in G.nodes(data=True)
           if isinstance(d.get("position"), (list, tuple)) and len(d["position"]) == 2]
    if pos:
        centroids.append(Point(np.mean(pos, axis=0)))
    else:
        geoms = [d.get("geometry") for _, d in G.nodes(data=True) if d.get("geometry")]
        xs = [g.x for g in geoms]; ys = [g.y for g in geoms]
        centroids.append(Point(np.mean(xs), np.mean(ys)))

centroid_gdf = gpd.GeoDataFrame(
    {"subgraph_index": range(len(clean_subgraphs))},
    geometry=centroids,
    crs="EPSG:28992"
)

cbs_pc6 = gpd.read_file(CBS_PC6_FILE)[["postcode6", "geometry"]]
if cbs_pc6.crs != centroid_gdf.crs:
    cbs_pc6 = cbs_pc6.to_crs(centroid_gdf.crs)

centroid_pc6 = gpd.sjoin_nearest(
    centroid_gdf, cbs_pc6,
    how="left", distance_col="distance"
)[["subgraph_index", "postcode6"]]

buurt = pd.read_csv(BUURT_LOOKUP)
buurt["postcode6"] = (
    buurt["postcode6"]
    .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    .apply(lambda x: x if isinstance(x, (list, tuple)) else [x])
)
expanded = (
    buurt.explode("postcode6")
    .assign(postcode6=lambda df: df["postcode6"].astype(str))
    .rename(columns={"buurtcode": "buurt"})
    [["buurt", "postcode6"]]
)

mapping = centroid_pc6.merge(expanded, on="postcode6", how="left")
mapping.to_csv(STORE_DIR / "subgraph_to_buurt.csv", index=False)

node_counts = defaultdict(list)
for _, row in mapping.dropna(subset=["buurt"]).iterrows():
    idx = int(row["subgraph_index"])
    node_counts[row["buurt"]].append(len(clean_subgraphs[idx]))
with open(STORE_DIR / "subgraphs_node_counts.json", "w", encoding="utf-8") as jf:
    json.dump(node_counts, jf, indent=2, ensure_ascii=False)

print("[Phase A] mapping + JSON written.")

# --- Phase B: save pickles by buurt or by node-count clusters ---
if STORE_MODE == "buurt":
    mapping = pd.read_csv(STORE_DIR / "subgraph_to_buurt.csv")
    grouped = defaultdict(list)
    for _, row in mapping.dropna(subset=["buurt"]).iterrows():
        idx = int(row["subgraph_index"])
        grouped[row["buurt"]].append(clean_subgraphs[idx])

    cons_df = pd.read_csv(CONS_DATA_FILE)
    metrics_json = {}
    saved = 0

    for buurt_code, subgs in tqdm(grouped.items(), desc="Phase B (buurt)"):
        total_nodes = sum(len(sg.nodes) for sg in subgs)
        neigh = cons_df[cons_df["CBS Buurtcode"] == buurt_code]
        if neigh.empty:
            continue

        sja  = neigh["kleinverbruik_SJA_GEMIDDELD"].iloc[0]
        met  = neigh["opwek_klein_AANTAL_AANSLUITINGEN_MET_OPWEKINSTALLATIE"].iloc[0]
        tot  = neigh["opwek_klein_AANTAL_AANSLUITINGEN_IN_CBS_BUURT"].iloc[0]
        aans = neigh["kleinverbruik_AANSLUITINGEN_AANTAL"].iloc[0]
        pct  = (met + 1) / (tot + 1) if tot >= 0 else 0

        groot_cols = [
            "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_WIND_IN_AANLEG",
            "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_WIND_IN_BEDRIJF",
            "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_ZON_IN_AANLEG",
            "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_ZON_IN_BEDRIJF"
        ]
        groot_sum = neigh[groot_cols].sum(axis=1).fillna(0).iloc[0]

        hours, kw_mw = 8760, 1000
        total_kwh = sja * aans
        avg_cons   = (total_kwh / hours) / kw_mw
        avg_gen    = (groot_sum / hours) / kw_mw
        avg_per_node = avg_cons / (total_nodes + 1e-6)

        buurt_data = {
            'subgraphs': subgs,
            'info': {
                'total_nodes': total_nodes,
                'nodes_per_subgraph': [len(sg.nodes) for sg in subgs],
                'percentage_gen_small': pct,
                'avg_power_mw_generation_groot': avg_gen,
                'avg_power_mw_consumption_small_per_node': avg_per_node,
                'original_opwek_groot_kwh_yearly': groot_sum,
                'original_ja_kwh_buurt_small': total_kwh
            }
        }

        fname = STORE_DIR / f"{buurt_code}_n-{len(subgs)}.pkl"
        with open(fname, 'wb') as out:
            pickle.dump(buurt_data, out, protocol=pickle.HIGHEST_PROTOCOL)

        metrics_json[buurt_code] = [len(sg.nodes) for sg in subgs]
        saved += 1

    with open(STORE_DIR / "subgraphs_node_counts_updated.json", 'w', encoding='utf-8') as jf:
        json.dump(metrics_json, jf, indent=2, ensure_ascii=False)

    print(f"[Phase B] buurt mode done. Saved {saved} files.")

elif STORE_MODE == "node":
    def chunker(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    buckets = defaultdict(list)
    for idx, G in enumerate(clean_subgraphs):
        n = G.number_of_nodes()
        lower = (n // NODE_CLUSTER_SZ) * NODE_CLUSTER_SZ
        upper = lower + NODE_CLUSTER_SZ
        label = f"n{str(lower).zfill(2)}_{str(upper).zfill(2)}"
        buckets[label].append(idx)

    for label, idx_list in buckets.items():
        for part, chunk in enumerate(chunker(idx_list, MAX_PER_CLUSTER), start=1):
            subgs = [clean_subgraphs[i] for i in chunk]
            fname = STORE_DIR / f"subgraph-{label}-{part}-n{len(subgs)}.pkl"
            with open(fname, 'wb') as out:
                pickle.dump(subgs, out, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Phase B] node mode done. {len(buckets)} buckets written.")

else:
    raise ValueError(f"Unknown STORE_MODE={STORE_MODE!r}")
"""