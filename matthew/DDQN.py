"""
Using double deep Q networks with target networks.
Size of agent is proportional to speed, eating food increases size and speed.
Using discounted utilities with warm starts for fairness rewards
"""
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
from utils import SI_reward, variance_penalty, get_fairness_from_su, EpsilonDecay, get_metrics_from_rewards, add_epi_metrics_to_logs, add_metric_to_logs
from Environment import MatthewEnvt


# Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

greedy = False
training = True
reallocate = False
central_rewards = False
simple_obs = False
logging = True

split = True
learn_fairness = True
learn_utility = True
multi_head = True

phased_training = False
phase_length = 200
if phased_training and not split:
	print("Phased training only supported for split")
	exit()
if phased_training and not (learn_fairness and learn_utility):
	print("Phased training only supported for learning both fairness and utility")
	exit()

if multi_head and not split:
	print("Multi head only supported for split")
	exit()
if multi_head and not (learn_fairness and learn_utility):
	print("Multi head only supported for learning both fairness and utility")
	exit()

SI_beta = 0
learning_beta = 0.011
fairness_type = "split_diff" # ['split_diff', 'variance_diff', 'split_variance', 'variance', 'SI']
# fairness_type = "variance_diff"

mode = "Reallocate" if reallocate else "Fixed"
mode += "Central" if central_rewards else ""
mode += "" if simple_obs else "Complex"
mode+= "/Test/" #Only for testing new features
network_type = ""
if multi_head:
	network_type = "/MultiHead"
elif split and not multi_head: 
	network_type = "/Split"
elif not split:
	network_type = "/Joint"
mode+= network_type
mode += "Phased" if phased_training else ""
if split and not learn_utility:
	mode += "NoUtility"
if split and not learn_fairness:
	mode += "NoFairness"
mode += "/"
mode += f"{fairness_type}"
mode += f"/{learning_beta}"

st_time = time.time()
if training and logging:
	# Create a summary writer
	log_dir = f"logs/DDQN/{mode}/{int(st_time)}/"
	print("Logging to {} \n\n\n\n".format(log_dir))
	summary_writer = tf.summary.create_file_writer(log_dir)
else:
	summary_writer = None

max_size = 0.50
n_agents=10
n_resources=3

warm_start = 50
past_discount = 0.995
#initialize environments
M = MatthewEnvt(n_agents=n_agents, n_resources=n_resources, max_size=max_size, reallocate=reallocate, simple_obs=simple_obs, warm_start=warm_start, past_discount=past_discount)
M_train = MatthewEnvt(n_agents=n_agents, n_resources=n_resources, max_size=max_size, reallocate=reallocate, simple_obs=simple_obs, warm_start=warm_start, past_discount=past_discount)
M_val = MatthewEnvt(n_agents=n_agents, n_resources=n_resources, max_size=max_size, reallocate=reallocate, simple_obs=simple_obs, warm_start=warm_start, past_discount=past_discount)

config = tf.compat.v1.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.compat.v1.Session(config=config)
KTF.set_session(session)
T = 100   # learning frequency for the policy and value networks
totalTime = 0
GAMMA = 0.98
n_episode = 15000
max_steps = 500
i_episode = 0
render = False

eps = EpsilonDecay(start=0.5, end=0.05, decay_rate=0.995, greedy=greedy)
ep_epsilon = eps.reset()

obs = M.get_obs()
num_features = len(obs[0][0])

# model_loc = "Models/DDQN/FixedComplex/0/1689619731/best/best_model.ckpt"
learning_rate = 0.00003
agent = DDQNAgent(M_train, num_features, hidden_size=256, learning_rate=learning_rate, replay_buffer_size=250000, GAMMA=GAMMA, learning_beta=learning_beta)
if split:
	if multi_head:
		agent = MultiHeadDDQNAgent(M_train, num_features, hidden_size=256, learning_rate=learning_rate, replay_buffer_size=250000, GAMMA=GAMMA, learning_beta=learning_beta,
			    learn_utility=learn_utility, learn_fairness=learn_fairness, phased_learning=phased_training)
	else:
		agent = SplitDDQNAgent(M_train, num_features, hidden_size=256, learning_rate=learning_rate, replay_buffer_size=250000, GAMMA=GAMMA, learning_beta=learning_beta,
				learn_utility=learn_utility, learn_fairness=learn_fairness)
		if not learn_utility:
			u_model_loc = "Models/DDQN/FixedComplex/Split/split_diff/0.0/1691557008/best/best_model.ckpt_util"
			agent.load_util_model(u_model_loc)
		if not learn_fairness:
			f_model_loc = "Models/DDQN/FixedComplex/0/"
			agent.load_fair_model(f_model_loc)
	
if not training:
	model_loc = ""
	agent.load_model(model_loc)

best_val_objective = -100000.0
run_metrics = {'utility':[], 'fairness':[], 'min_utility':[], 'objective':[],'variance':[]}
while i_episode<n_episode:
	i_episode+=1

	VF1_loss = []
	VF2_loss = []
	FF1_loss = []
	FF2_loss = []

	score=0  #Central agent score = sum of agent scores
	steps=0
	
	M.reset()
	obs = M.get_obs()

	ep_epsilon = eps.decay()

	selected_VF_id = random.randint(0,1)
	print("Selected VF", selected_VF_id)
	agent.set_active_net(selected_VF_id)
	while steps<max_steps:

		steps+=1
		#For each agent, select the action: central allocation
		# actions = compute_best_actions(agent, obs, M.targets, n_agents, n_resources, M.discounted_su, beta=SI_beta, epsilon=ep_epsilon) # Deprecated
		actions = M.compute_best_actions(agent, M, obs, beta=SI_beta, epsilon=ep_epsilon) # New way of computing actions
		pd_states = M.get_post_decision_states(obs, actions)
		
		su_prev = copy.deepcopy(M.discounted_su)
		#Take a step based on the actions, get rewards and updated agents, resources etc.
		rewards = M.step(actions)
		score += sum(rewards)
		obs = M.get_obs()
		su_post = copy.deepcopy(M.discounted_su)
		f_rewards = get_fairness_from_su(su_prev, su_post, ftype=fairness_type, action=actions)

		#Add to replay buffer
		#Experience stores - [(s,a), r(s,a,s'), r_f(s,a,s'), s']
		done = steps==max_steps
		agent.add_experience(pd_states, rewards, f_rewards, M.get_state(), done)

		#Update the policies
		if steps%T==0 and training:
			if split:
				loss_logs = agent.update(num_samples=32)
				if learn_fairness and len(loss_logs['fair'])>0:
					FF1_loss.extend(loss_logs['fair'][0])
					FF2_loss.extend(loss_logs['fair'][1])
				if learn_utility and len(loss_logs['util'])>0:
					VF1_loss.extend(loss_logs['util'][0])
					VF2_loss.extend(loss_logs['util'][1])
			else:
				losses1, losses2 = agent.update(num_samples=32)
				VF1_loss.extend(losses1)
				VF2_loss.extend(losses2)
		
		if render:
			M.render()
			time.sleep(0.01)
	
	losses_dict = None
	if training:
		losses_dict = {"VF1":np.mean(VF1_loss), "VF2":np.mean(VF2_loss), "Value_Loss": np.mean(VF1_loss+VF2_loss)}
		if split:
			losses_dict["FF1"] = np.mean(FF1_loss)
			losses_dict["FF2"] = np.mean(FF2_loss)
			losses_dict["Fair_Loss"] = np.mean(FF1_loss+FF2_loss)
			
	epi_metrics = add_epi_metrics_to_logs(summary_writer, M.su, losses_dict, learning_beta, i_episode, max_steps, verbose=True, prefix="", logging=logging)
	for key, value in epi_metrics.items():
		run_metrics[key].append(value)
		#Print the average metrics
		if i_episode%50==0:
			print("Average "+key+": ", np.mean(run_metrics[key]))

	if training:
		#update the target network every 20 episodes
		if i_episode%20==0:
			agent.update_target_networks()
		
		#Switch the phase every phase_length episodes
		if phased_training and i_episode%phase_length==0:
			agent.switch_phase()
			print("Switched phase")

	# Save the model every 500 episodes
	if i_episode%500==0:
		os.makedirs(f"Models/DDQN/{mode}/{int(st_time)}/", exist_ok=True)
		agent.save_model(f"Models/DDQN/{mode}/{int(st_time)}/model_{i_episode}.ckpt")

	# # Validation runs every 500 episodes, to select best model
	if i_episode%10==0 and training:
		#set selected VF to 0
		agent.set_active_net(0)
		print("Validating")
		update = i_episode%100==0
		mult = 25 if update else 1

		#Run validation episodes with the current policy
		val_metrics = {'utility':[], 'fairness':[], 'min_utility':[], 'objective':[],'variance':[]}
		for val_eps in range(1*mult):
			print(val_eps, end='\r')
			M_val.reset()
			obs = M_val.get_obs()
			score = 0
			for steps in range(max_steps):
				actions = M_val.compute_best_actions(agent, M_val, obs, beta=SI_beta, epsilon=0)
				rewards = M_val.step(actions)
				score += sum(rewards)
				obs = M_val.get_obs()
			print(M_val.su)
			metrics = get_metrics_from_rewards(M_val.su, learning_beta)
			for key, value in metrics.items():
				val_metrics[key].append(value)
		mean_val_metrics = {}
		for key, value in val_metrics.items():
			mean_val_metrics[key] = np.mean(value)
			add_metric_to_logs(summary_writer, np.mean(value), "Validation_"+key, i_episode, logging=logging, verbose=True)

		if  update and best_val_objective<mean_val_metrics['objective'] and i_episode>1000: #At least 1000 episodes before trying to save a best model.
			best_val_objective = mean_val_metrics['objective']
			#make directory if it doesn't exist
			os.makedirs(f"Models/DDQN/{mode}/{int(st_time)}/best", exist_ok=True)
			agent.save_model(f"Models/DDQN/{mode}/{int(st_time)}/best/best_model.ckpt")
			print("Saved best model")
			#Write the logs to a file
			with open(f"Models/DDQN/{mode}/{int(st_time)}/best/best_log.txt", "w") as f:
				f.write(f"Validation Utility: {mean_val_metrics['utility']}\n")
				f.write(f"Validation Fairness: {mean_val_metrics['fairness']}\n")
				f.write(f"Validation Min Utility: {mean_val_metrics['min_utility']}\n")
				f.write(f"Validation Objective: {mean_val_metrics['objective']}\n")
				f.write(f"Validation Variance: {mean_val_metrics['variance']}\n")
				#Write the episode number, epsilon, simple_obs, reallocate, central_rewards
				f.write(f"Episode: {i_episode}\n")
				f.write(f"Epsilon: {ep_epsilon}\n")
				f.write(f"Simple Obs: {simple_obs}\n")
				f.write(f"Reallocate: {reallocate}\n")
				f.write(f"Central Rewards: {central_rewards}\n")
				f.write(f"SI Beta: {SI_beta}\n")
				f.write(f"Learning Beta: {learning_beta}\n")
				f.write(f"Learning Rate: {learning_rate}\n")
				f.write(f"Fairness Type: {fairness_type}\n")
				f.write(f"Warm Start: {warm_start}\n")
				f.write(f"Past Discount: {past_discount}\n")
				if not learn_utility:
					f.write(f"Utility Model: {u_model_loc}\n")
				f.write(f'Multi Head: {multi_head}\n')
				f.write(f"Phased Training: {phased_training}\n")
				f.write(f"Phase Length: {phase_length}\n")
