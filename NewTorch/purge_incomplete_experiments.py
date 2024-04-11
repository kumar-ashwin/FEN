# Takes a directory and removes all incomplete experiments
# Complete experiments are identified by the presence of a results.csv file
# Only use in the NewTorch context.
# Given a top level directory containing multiple experiments for different beta values, this script opens all of them, looks through all bootstraps (folders) and removes any that do not contain a results.csv file

import os
expt_dir = "logs/WarmStart/Simple/split_diff/Joint/"

for beta in os.listdir(expt_dir):
    beta_dir = os.path.join(expt_dir, beta)
    for bootstrap_id in os.listdir(beta_dir):
        bootstrap_dir = os.path.join(beta_dir, bootstrap_id)
        if not os.path.exists(os.path.join(bootstrap_dir, "results.csv")):
            print(f"Removing {bootstrap_dir}")
            # recursively delete all files in the directory
            print(f"rm -rf {bootstrap_dir}")
            os.system(f"rm -rf {bootstrap_dir}")