## Overview
This repository contains the full pipeline to devevelop graph neural networks to solve distribution network reconfiguration. This repo can in general be subdivided into two components

0. *link to paper*
1. *Data Generation*
2. *GNN Development*

---

##  Link to paper
[Graph Neural Networks for Distribution Network Reconfiguration Optimization - Dennis Hollanders](https://research.tue.nl/nl/studentTheses/graph-neural-networks-for-distribution-network-reconfiguration-op) 

---

## 1. Data Generation
*Foldername:* data_generation

### Purpose
- Generate a folder with pandapower networks that serve as input to the optimization
- The data can be generated in two ways:
 1. Synthetic generation based on dutch cable locations
 2. Sample datasets from pandapower/simbench libraries
- After this the generate_ground_truth file can be used to apply an MISOCP optimization on the generated data which serves as ground truth for the supervised GNN methods.

### Key sections 
- data_generation/create_dataset.py:
- data_generation/define_ground_truth.py
- src/electrify_subgraphs.py:
- src/createtestval.py:
- src/SOCP_class_dnr.py:

### Example
Below a snippet of code is provided to illustrate how to generate a dataset through CLI. 


*Step 1. Init Repo* 

```ruby
git clone https://github.com/DennisHollanders/gnn-dnr/
cd gnn-dnr
poetry init 
```

*Step 2. Create Dataset* 

```
poetry run python data_generation/create_dataset.py --generate_synthetic_data False --sample_real_data True --dataset_names ["train","validation","test"] --samples_per_dataset [100,10,10]
```

*Step 3: Generate Ground Truth*

```
poetry run python data_generation/define_ground_truth --folder_path  <your-root>/data/<name-dataset>/train --debug False
```

---


## 2. GNN development 
*Foldername:* model_search

### Purpose
- Create a GNN model
- Apply that GNN model within one of the warmstart frameworks
- Experimentation with GNN + embedded optimization through cvxpy

### Key sections 
- model_search/main.py
- model_search/predict_then_optimize.py
- model_search/hpo.py

- models:
    1. model_search/models/AdvcanedMLP (GNNs)
    2. model_search/models/cvx

- config files:
    1. model_search/config-mlp-GCN.yaml
    2. model_search/config_files/config-cvx.yaml

### Example
for these examples first adapt the config files, which are stored in model_search/config_files, to reference to the generated dataset paths.

*Example 1: Train GNN*
```
poetry run python model_search/main.py --config config-mlp.yaml 
```

*Example 2: Run HPO for GAT*
```
poetry run python model_search/hpo.py --config hpo-GAT.yaml 
```

*Example 3: Train Decision-focused model*

```
poetry run python model_search/main.py --config config-cvx.yaml 
```
--- 

*Example 4: Evaluate Predict Then Optimize Model*

```
poetry run python model_search/predict_then_optimize.py --folder_path <insert folder_path> --model_path <insert trained model path> --mode soft
```
--- 