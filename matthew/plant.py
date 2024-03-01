import os, sys, time  
import numpy as np
import tensorflow as tf
import random

from tensorflow.python.keras import backend as KTF
from keras import backend as K
import copy
import matplotlib.pyplot as plt

from tensorboard.plugins.hparams import api as hp

from Agents import DDQNAgent, SplitDDQNAgent, MultiHeadDDQNAgent
from matching import compute_best_actions
from agent_utils import get_agent, take_env_step, training_step, post_episode_hk
from utils import SI_reward, variance_penalty, get_fairness_from_su, EpsilonDecay, get_metrics_from_rewards, add_epi_metrics_to_logs, add_metric_to_logs
from Environment import PlantEnvt
from process_args import process_args

# Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

args, train_args = process_args(env_name='Plant')

if args.training and args.logging:
	log_dir = f"logs/{args.save_path}/"
	print("Logging to {} \n\n\n\n".format(log_dir))
	summary_writer = tf.summary.create_file_writer(log_dir)
else:
	summary_writer = None

# Env params
n_agents = 5
n_resources = 8
gridsize = 12
M = PlantEnvt(n_agents=n_agents, gridsize=gridsize, n_resources=n_resources, reallocate=args.reallocate, simple_obs=args.simple_obs, warm_start=args.warm_start, past_discount=args.past_discount)
M_train = copy.deepcopy(M)
M_val = copy.deepcopy(M)

i_episode = 0 #episode counter
eps = EpsilonDecay(start=0.9, end=0.05, decay_rate=0.995, greedy=args.greedy)
ep_epsilon = eps.reset()

obs = M.get_obs()
num_features = len(obs[0][0])

agent = get_agent(train_args, args.training, num_features, M_train)

best_val_objective = -100000.0
run_metrics = {'utility':[], 'fairness':[], 'min_utility':[], 'objective':[],'variance':[]}

while i_episode<args.n_episode:
	i_episode+=1
	losses = {"FF1": [], "FF2": [], "VF1": [], "VF2": []}

	score=0  #Central agent score = sum of agent scores
	steps=0
	
	M.reset()
	obs = M.get_obs()

	ep_epsilon = eps.decay()

	selected_VF_id = random.randint(0,1)
	print("Selected VF", selected_VF_id)
	agent.set_active_net(selected_VF_id)

	while steps<args.max_steps:
		steps+=1
		#For each agent, select the action: central allocation
		use_greedy = True if i_episode<100 else False
		util, obs = take_env_step(M, agent, obs, steps, ep_epsilon, args, use_greedy=use_greedy)
		score += util
		
		#Update the policies
		if steps%train_args.model_update_freq ==0 and args.training:
			training_step(agent, train_args, losses)
		
		if args.render:
			M.render(agent)
			time.sleep(0.1)
	losses_dict = post_episode_hk(agent, losses, i_episode, train_args, args)
	epi_metrics = add_epi_metrics_to_logs(summary_writer, M.su, losses_dict, args.learning_beta, i_episode, args.max_steps, verbose=True, prefix="", logging=args.logging)
	if args.training:
		print("VF Loss", losses_dict["Value_Loss"])
		print("Fair Loss", losses_dict["Fair_Loss"])
	
	for key, value in epi_metrics.items():
		run_metrics[key].append(value)
		#Print the average metrics
		if i_episode%50==0:
			print("Average "+key+": ", np.mean(run_metrics[key]))

	# # Validation runs to select best model
	if i_episode%train_args.validation_freq==0 and args.training:
		#set selected VF to 0
		agent.set_active_net(0)
		print("Validating")
		update = i_episode%train_args.best_model_update_freq==0
		mult = 25 if update else 1

		#Run validation episodes with the current policy
		val_metrics = {'utility':[], 'fairness':[], 'min_utility':[], 'objective':[],'variance':[]}
		for val_eps in range(1*mult):
			M_val.reset()
			obs = M_val.get_obs()
			score = 0
			for steps in range(args.max_steps):
				util, obs = take_env_step(M_val, agent, obs, steps, 0, args, add_to_replay=False)
				score+=util

			print(M_val.su)
			metrics = get_metrics_from_rewards(M_val.su, args.learning_beta)
			for key, value in metrics.items():
				val_metrics[key].append(value)
		
		mean_val_metrics = {}
		for key, value in val_metrics.items():
			mean_val_metrics[key] = np.mean(value)
			add_metric_to_logs(summary_writer, np.mean(value), "Validation_"+key, i_episode, logging=args.logging, verbose=True)

		if  update and best_val_objective<mean_val_metrics['objective'] and i_episode>500: #At least 500 episodes before trying to save a best model.
			best_val_objective = mean_val_metrics['objective']
			#make directory if it doesn't exist
			os.makedirs(f"Models/{args.save_path}/best", exist_ok=True)
			agent.save_model(f"Models/{args.save_path}/best/best_model.ckpt")
			print("Saved best model")
			#Write the logs to a file
			with open(f"Models/{args.save_path}/best/best_log.txt", "w") as f:
				f.write(f"Validation Utility: {mean_val_metrics['utility']}\n")
				f.write(f"Validation Fairness: {mean_val_metrics['fairness']}\n")
				f.write(f"Validation Min Utility: {mean_val_metrics['min_utility']}\n")
				f.write(f"Validation Objective: {mean_val_metrics['objective']}\n")
				f.write(f"Validation Variance: {mean_val_metrics['variance']}\n")
				#Write the episode number, epsilon, simple_obs, reallocate, central_rewards
				f.write(f"Episode: {i_episode}\n")
				f.write(f"Epsilon: {ep_epsilon}\n")
				f.write("Arguments:\n")
				for arg in vars(args):
					f.write(f"{arg}: {getattr(args, arg)}\n")
				f.write("Training Arguments:\n")
				for arg in vars(train_args):
					f.write(f"{arg}: {getattr(train_args, arg)}\n")

import csv
# Final round of validation and saving results
if args.training:
	#set selected VF to 0
	agent.set_active_net(0)
	print("Final Validation")
	num_eps = 50
	M_val.external_trigger = True


	#Run validation episodes with the current policy
	val_metrics = {'utility':[], 'fairness':[], 'min_utility':[], 'objective':[],'variance':[]}
	for val_eps in range(num_eps):
		M_val.reset()
		obs = M_val.get_obs()
		score = 0
		for steps in range(args.max_steps):
			util, obs = take_env_step(M_val, agent, obs, steps, 0, args, add_to_replay=False)
			score+=util

			if args.render:
				M_val.render()
				time.sleep(0.1)
				print(steps, val_eps)

		if score==0:
			print("Zero score")
			M_val.render()

		print(M_val.su)
		metrics = get_metrics_from_rewards(M_val.su, args.learning_beta)
		for key, value in metrics.items():
			val_metrics[key].append(value)
	
	mean_val_metrics = {}
	for key, value in val_metrics.items():
		mean_val_metrics[key] = np.mean(value)
		# Results file is a csv
	# If the file doesn't exist, create it and write the header
	# Create the directory if it doesn't exist
	results_file = f"Results/{args.env_name}results.csv"
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
	

			