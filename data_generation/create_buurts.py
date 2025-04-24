#!/usr/bin/env python3
"""
process_subgraphs.py

Two‐phase pipeline:
  Phase A: split `filtered_subgraphs.pkl` into subgraph→buurt mapping and node‐count JSON
  Phase B: load mapping, compute metrics per buurt, write per‐buurt pickles and updated JSON

Usage:
  python process_subgraphs.py
"""
import pickle
from pathlib import Path
import ast
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm
import json
from collections import defaultdict

# --- Configuration ---
DATA_DIR      = Path.cwd().parent / "data"
SUBGRAPH_FILE = DATA_DIR / "filtered_subgraphs.pkl"
STORE_DIR     = DATA_DIR / "cbs_buurts"
STORE_DIR.mkdir(parents=True, exist_ok=True)

# --- Phase A: split and assign ---
# 1) Load all subgraphs
with open(SUBGRAPH_FILE, "rb") as f:
    all_subgraphs = pickle.load(f)

# 2) Compute centroids
centroids = []
for sg in tqdm(all_subgraphs, desc="Phase A: centroids"):
    pos = [d["position"] for _, d in sg.nodes(data=True)
           if "position" in d and isinstance(d["position"], (list, tuple)) and len(d["position"]) == 2]
    if pos:
        centroids.append(Point(np.mean(pos, axis=0)))
    else:
        geoms = [d.get("geometry") for _, d in sg.nodes(data=True) if d.get("geometry")]
        xs = [g.x for g in geoms]; ys = [g.y for g in geoms]
        centroids.append(Point(np.mean(xs), np.mean(ys)))

centroid_gdf = gpd.GeoDataFrame(
    {"subgraph_index": list(range(len(all_subgraphs)))},
    geometry=centroids,
    crs="EPSG:28992"
)

# 3) Load CBS postcode6 and project
cbs_pc6 = gpd.read_file(DATA_DIR / "cbs_pc6_2023.gpkg")[["postcode6","geometry"]]
if cbs_pc6.crs != centroid_gdf.crs:
    cbs_pc6 = cbs_pc6.to_crs(centroid_gdf.crs)

# 4) Nearest‐neighbor join
centroid_pc6 = gpd.sjoin_nearest(
    centroid_gdf, cbs_pc6,
    how="left", distance_col="distance"
)[["subgraph_index","postcode6"]]

# 5) Explode buurt lookup
buurt = pd.read_csv(DATA_DIR / "buurt_to_postcodes.csv")
buurt["postcode6"] = (
    buurt["postcode6"]
    .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    .apply(lambda x: x if isinstance(x, (list, tuple)) else [x])
)
expanded = (
    buurt.explode("postcode6")
    .assign(postcode6=lambda df: df["postcode6"].astype(str))
    .rename(columns={"buurtcode":"buurt"})
    [["buurt","postcode6"]]
)

# 6) Merge mapping
mapping = centroid_pc6.merge(expanded, on="postcode6", how="left")
# Save mapping CSV
mapping.to_csv(STORE_DIR / "subgraph_to_buurt.csv", index=False)
# Save node counts JSON
node_counts = defaultdict(list)
for _, row in mapping.dropna(subset=["buurt"]).iterrows():
    idx = int(row["subgraph_index"])
    node_counts[row["buurt"]].append(len(all_subgraphs[idx].nodes))
with open(STORE_DIR / "subgraphs_node_counts.json", "w", encoding="utf-8") as jf:
    json.dump(node_counts, jf, indent=2, ensure_ascii=False)

print("[Phase A] mapping + JSON written.")

# --- Phase B: compute metrics & save per‐buurt pickles ---
# Load mapping
mapping = pd.read_csv(STORE_DIR / "subgraph_to_buurt.csv")
# Rebuild clusters
grouped = defaultdict(list)
for _, row in mapping.dropna(subset=["buurt"]).iterrows():
    idx = int(row["subgraph_index"])
    grouped[row["buurt"]].append(all_subgraphs[idx])

# Load consumption data
cons_df = pd.read_csv(DATA_DIR / "aggregated_kleinverbruik_with_opwek.csv")

metrics_json = {}
saved = 0
for buurt_code, subgs in tqdm(grouped.items(), desc="Phase B: saving"):
    total_nodes = sum(len(sg.nodes) for sg in subgs)
    neigh = cons_df[cons_df["CBS Buurtcode"] == buurt_code]
    if neigh.empty:
        continue

    sja = neigh["kleinverbruik_SJA_GEMIDDELD"].iloc[0]
    met = neigh["opwek_klein_AANTAL_AANSLUITINGEN_MET_OPWEKINSTALLATIE"].iloc[0]
    tot = neigh["opwek_klein_AANTAL_AANSLUITINGEN_IN_CBS_BUURT"].iloc[0]
    aans = neigh["kleinverbruik_AANSLUITINGEN_AANTAL"].iloc[0]
    pct = (met + 1) / (tot + 1) if tot >= 0 else 0

    groot_cols = [
        "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_WIND_IN_AANLEG",
        "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_WIND_IN_BEDRIJF",
        "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_ZON_IN_AANLEG",
        "opwek_groot_MAXIMUM_OMVORMER_CAPACITEIT_ZON_IN_BEDRIJF"
    ]
    groot_sum = neigh[groot_cols].sum(axis=1).fillna(0).iloc[0]

    hours, kw_mw = 8760, 1000
    total_kwh = sja * aans
    avg_cons = (total_kwh / hours) / kw_mw
    avg_gen = (groot_sum / hours) / kw_mw
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

    fname = f"{buurt_code}_n-{len(subgs)}.pkl"
    with open(STORE_DIR / fname, 'wb') as out:
        pickle.dump(buurt_data, out)
    saved += 1
    metrics_json[buurt_code] = [len(sg.nodes) for sg in subgs]

# Save updated metrics JSON
with open(STORE_DIR / "subgraphs_node_counts_updated.json", 'w', encoding='utf-8') as jf:
    json.dump(metrics_json, jf, indent=2, ensure_ascii=False)

print(f"[Phase B] done. Saved {saved} buurten.")
