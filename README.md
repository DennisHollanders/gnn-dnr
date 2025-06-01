This repository contains the full pipeline for training a graph neural network to perform distribution network reconfiguration. -

# Content
1. data_generation
          - create_dataset:
          - create_gt:
2. model_search
          - models folder
          - configs folder
          - train/load_data/evaluation/preprocess_data
3. How to use
4. Hardware used

# 1. data_generation

# 2. model_search

# 3. How to use
- put required data in "data" folder
- init poetry
- poetry run python data_generation/create_dataset --....
- poetry run python data_generation/create_gt --.... 

# 4. Hardware used
