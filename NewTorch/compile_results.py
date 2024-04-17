# Takes a directory and compiles all results
# Complete experiments are identified by the presence of a results.csv file
# Only use in the NewTorch context.
# Given a top level directory containing multiple experiments for different beta values, this script opens all of them, looks through all bootstraps (folders) and aggregates the results into a single CSV file.

import os

base_dir = "logs/WarmStart/Simple/split_diff/"
save_name = "WarmStart"
results_file = "Results/Compiled/"+save_name+".csv"

import csv
import pandas as pd
import numpy as np

all_results = []
headers = None
for expts in os.listdir(base_dir):
	expt_dir = os.path.join(base_dir, expts)
	for beta in os.listdir(expt_dir):
		print(expts, beta)
		beta_dir = os.path.join(expt_dir, beta)
		for bootstrap_id in os.listdir(beta_dir):
			bootstrap_dir = os.path.join(beta_dir, bootstrap_id)
			if not os.path.exists(os.path.join(bootstrap_dir, "results.csv")):
				continue

			with open(os.path.join(bootstrap_dir, "results.csv"), 'r') as f:
				#check if it has a header
				expt_row =  pd.read_csv(f)
				if len(expt_row) == 0:
					print(expt_row)
					#try reading without header
					expt_row = pd.read_csv(bootstrap_dir+"/results.csv", header=None)
				else: 
					if headers is None:
						headers = expt_row.columns

				#add to all results as dict
				all_results.append(expt_row)

# Add headers to all results
if headers is not None:
	for i, expt in enumerate(all_results):
		expt.columns = headers

#concatenate all results
all_results = pd.concat(all_results)

#Re-order columns, to bring learning beta, split, multi_head and learn_utility to the front
cols = all_results.columns.tolist()
cols_set = set(cols)
front_cols = ['learning_beta', 'split', 'multi_head', 'learn_utility', 'learn_fairness']
for col in cols_set:
	if col in front_cols:
		cols.remove(col)

new_cols = front_cols + cols
all_results = all_results[new_cols]
#save to csv
#make sure the directory exists
os.makedirs(os.path.dirname(results_file), exist_ok=True)
all_results.to_csv(results_file, index=False)