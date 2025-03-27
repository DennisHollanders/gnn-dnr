1. Initialize poetry env
2. Add data zip to a data folder:
          path structure: data\test_val_real__range-30-150_nTest-5_nVal-5_2732025_1

3. poetry run python data_generation/define_ground_truth.py
4. Optionally poetry run python data_generation/define_ground_truth.py --include_radiality_constraints True --use_spanning_tree_radiality False

Note:  I tried adding slack variables as mentioned yesterday, I however kept the original code in comments.
Note: The last network fails due to some indexing errors you can ignore this, will check this myself later. 
