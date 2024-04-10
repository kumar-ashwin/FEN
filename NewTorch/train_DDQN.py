import os, time  
import numpy as np
import tensorflow as tf
import random
import csv

from agent_utils import get_agent, take_env_step, training_step, post_episode_hk, get_env, run_validation, save_best_model
from utils import EpsilonDecay, add_epi_metrics_to_logs, add_metric_to_logs
from process_args import process_args

# Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

args, train_args = process_args()

if args.training and args.logging:
	log_dir = f"{args.save_path}/"
	print("Logging to {} \n\n\n\n".format(log_dir))
	summary_writer = tf.summary.create_file_writer(log_dir)
else:
	summary_writer = None

# Env params
M, M_train, M_val = get_env(args.env_name, args, train_args)
M_val.external_trigger = True

eps = EpsilonDecay(start=0.9, end=0.05, decay_rate=0.98, greedy=args.greedy)
ep_epsilon = eps.reset()

obs = M.get_obs()
num_features = len(obs[0][0])

agent = get_agent(train_args, args.training, num_features, M_train)

best_val_objective = -100000.0
run_metrics = {'system_utility':[], 'fairness':[], 'min_utility':[], 'objective':[],'variance':[]}

# while i_episode<args.n_episode:
for i_episode in range(1, args.n_episode+1):
	ep_epsilon = eps.decay()
	losses = {"FF": [],  "VF": []}

	score=0  #Central agent score = sum of agent scores
	
	M.reset()
	obs = M.get_obs()

	# Run the episode
	for steps in range(1, args.max_steps+1):
		#For each agent, select the action: central allocation
		use_greedy = True if i_episode<0 else False
		util, obs = take_env_step(M, agent, obs, steps, ep_epsilon, args, use_greedy=use_greedy)
		score += util
		# print(M.posessions)
		
		#Update the policies
		if steps%train_args.model_update_freq ==0 and args.training:
			training_step(agent, train_args, losses)
		
		if args.render:
			M.render()
			time.sleep(0.1)
	
	# Post episode housekeeping
	losses_dict = post_episode_hk(agent, losses, i_episode, train_args, args)
	epi_metrics = add_epi_metrics_to_logs(summary_writer, M.su, losses_dict, args.learning_beta, i_episode, args.max_steps, verbose=True, prefix="", logging=args.logging, fair_rewards=M.get_fairness_rewards())
	print("Epsilon", ep_epsilon)
	if args.training:
		print("VF Loss", losses_dict["Value_Loss"])
		print("Fair Loss", losses_dict["Fair_Loss"])

	for key, value in epi_metrics.items():
		run_metrics[key].append(value)
		#Print the average metrics
		if i_episode%50==0:
			print("Average "+key+": ", np.mean(run_metrics[key]))

	# Validation runs to select best model
	if i_episode%train_args.validation_freq==0 and args.training:
		print("Validating")
		update = i_episode%train_args.best_model_update_freq==0
		num_epochs = 25 if update else 1

		val_metrics = run_validation(num_epochs, M_val, agent, args, render=args.render)
		# val_metrics = run_validation(num_epochs, M_val, agent, args, render=True)
		
		mean_val_metrics = {}
		for key, value in val_metrics.items():
			mean_val_metrics[key] = np.mean(value)
			add_metric_to_logs(summary_writer, np.mean(value), "Validation_"+key, i_episode, logging=args.logging, verbose=True)

		# Save the best model
		min_steps = 20
		if  update and best_val_objective<mean_val_metrics['objective'] and i_episode>=min_steps:
			best_val_objective = mean_val_metrics['objective']
			save_best_model(agent, mean_val_metrics, i_episode, ep_epsilon, args, train_args)

# Final round of validation and saving results
if args.training:
	#load best model
	if train_args.split and not train_args.multi_head:
		if train_args.learn_utility:
			agent.load_util_model(f"{args.save_path}/models/best/best_model.ckpt_util")
		if train_args.learn_fairness:
			agent.load_fair_model(f"{args.save_path}/models/best/best_model.ckpt_fair")
	else:
		agent.load_model(f"{args.save_path}/models/best/best_model.ckpt")
	print("Final Validation")
	num_eps = 50
	M_val.external_trigger = True

	val_metrics = run_validation(num_eps, M_val, agent, args, render=args.render) 
	# TODO: need to upate above function automatically take the relevant metric for fairness (reward vs fairness_reward)
	
	mean_val_metrics = {}
	for key, value in val_metrics.items():
		mean_val_metrics[key] = np.mean(value)
		# Results file is a csv
	# If the file doesn't exist, create it and write the header
	# Create the directory if it doesn't exist
	results_file = f"Results/{args.env_name+args.env_name_mod}results.csv"
	create=False
	if not os.path.exists("Results"):
		os.makedirs("Results")
	file_exists = os.path.exists(results_file)

	with open(results_file, "a", newline="") as f:	
		# Add one row to the csv file
		all_fields = {}
		for key, value in mean_val_metrics.items():
			all_fields[key] = value
		for arg in vars(args):
			all_fields[arg] = getattr(args, arg)
		for arg in vars(train_args):
			all_fields[arg] = getattr(train_args, arg)
		all_fields["observation_space"] = M.observation_space
		
		writer = csv.DictWriter(f, fieldnames=all_fields.keys())
		if not file_exists:
			writer.writeheader()
		writer.writerow(all_fields)

# Write a file to indicate that the training is done
if args.training:
	with open(f"{args.save_path}/training_done.txt", "w") as f:
		f.write("Training Done")