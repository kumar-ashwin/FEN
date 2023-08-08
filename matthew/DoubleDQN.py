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

from Agents import SimpleValueNetwork, ValueNetwork, ReplayBuffer, DDQNAgent, SplitDDQNAgent
from matching import compute_best_actions
from utils import SI_reward, variance_penalty, get_fairness_from_su, EpsilonDecay
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

split = False
learn_fairness = True
learn_utility = True

beta = 0
learning_beta = 0.5
fairness_type = "split_diff" # ['split_diff', 'split_variance', 'variance_diff', 'variance', 'SI']

max_size = 0.5
# if central_rewards:
# 	print("Central rewards for DQN not implemented yet. Exiting")
# 	exit()

mode = "Reallocate" if reallocate else "Fixed"
mode += "Central" if central_rewards else ""
mode += "" if simple_obs else "Complex"
# mode+= "/Split/" if split else "/Joint/"
mode += f"_{fairness_type}"
mode += f"/{learning_beta}"

st_time = time.time()
if training and logging:
	# Create a summary writer
	log_dir = f"logs/DoubleDQN/{mode}/{int(st_time)}/"
	print("Logging to {} \n\n\n\n".format(log_dir))
	summary_writer = tf.summary.create_file_writer(log_dir)

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
n_episode = 5000
max_steps = 500
i_episode = 0
render = False

eps = EpsilonDecay(start=0.5, end=0.05, decay_rate=0.995, greedy=greedy)
ep_epsilon = eps.reset()

obs = M.get_obs()
num_features = len(obs[0][0])

# model_loc = "Models/DoubleDQN/FixedComplex/0/1689619731/best/best_model.ckpt"
learning_rate = 0.00003
agent = DDQNAgent(M_train, num_features, hidden_size=256, learning_rate=learning_rate, replay_buffer_size=250000, GAMMA=GAMMA, learning_beta=learning_beta)
if split:
	agent = SplitDDQNAgent(M_train, num_features, hidden_size=256, learning_rate=learning_rate, replay_buffer_size=250000, GAMMA=GAMMA, learning_beta=learning_beta,
			learn_utility=learn_utility, learn_fairness=learn_fairness)
	if not learn_utility:
		u_model_loc = "Models/DoubleDQN/FixedComplex/0/"
		agent.load_util_model(u_model_loc)
	if not learn_fairness:
		f_model_loc = "Models/DoubleDQN/FixedComplex/0/"
		agent.load_fair_model(f_model_loc)
	
if not training:
	model_loc = ""
	agent.load_model(model_loc)

best_val_utility = 0.0
fairness = []
utility = []
mins = []
while i_episode<n_episode:
	i_episode+=1

	VF1_loss = []
	VF2_loss = []

	score=0  #Central agent score = sum of agent scores
	steps=0
	
	M.reset()
	obs = M.get_obs()

	ep_epsilon = eps.decay()

	selected_VF_id = random.randint(0,1)
	agent.set_active_net(selected_VF_id)
	while steps<max_steps:

		steps+=1
		#For each agent, select the action: central allocation
		actions = compute_best_actions(agent, obs, M.targets, n_agents, n_resources, M.discounted_su, beta=beta, epsilon=ep_epsilon)
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
		agent.add_experience(pd_states, rewards, f_rewards, M.get_state())

		#Update the policies
		if steps%T==0 and training:
			if not split:
				losses1, losses2 = agent.update(num_samples=32)
				VF1_loss.extend(losses1)
				VF2_loss.extend(losses2)
			else:
				print("TODO: Split update")
				# losses1, losses2 = agent.update(num_samples=32)
		
		if render:
			M.render()
			time.sleep(0.01)

	print(i_episode)
	print("Selected VF", selected_VF_id)
	print(score/max_steps) #Average reward
	print(M.su) #Agent rewards
	print(M.discounted_su, '(discounted)')
	uti = np.array(M.su)/max_steps
	print("Fairness",np.var(uti)/np.mean(uti)) #Fairness
	print('epsilon', ep_epsilon)
	if training:
		print("VF learning beta", learning_beta)

	fairness.append(np.var(uti)/np.mean(uti))
	utility.append(np.mean(score/max_steps))
	mins.append(min(M.su))


	print("Average Utility: ", np.mean(utility))
	print("Average Fairness: ", np.mean(fairness))
	print("Average Min Utility: ", np.mean(mins))

	if training and logging:
		with summary_writer.as_default():
			if len(VF1_loss)>0:
				tf.summary.scalar("Value_Loss 1", float(np.mean(VF1_loss)), step=i_episode)
				tf.summary.scalar("Value_Loss 2", float(np.mean(VF2_loss)), step=i_episode)
				tf.summary.scalar("Value_Loss", float(np.mean(VF1_loss+VF2_loss)), step=i_episode)
			tf.summary.scalar("Utility", float(score/max_steps), step=i_episode)
			tf.summary.scalar("Fairness", float(np.var(uti)/(np.mean(uti)+0.00001)), step=i_episode)
			tf.summary.scalar("Min_Utility", float(min(M.su)), step=i_episode)
		
		#update the target network every 20 episodes
		if i_episode%20==0:
			agent.update_target_networks()

	# Save the model every 1000 episodes
	if i_episode%1000==0:
		agent.save_model(f"Models/DoubleDQN/{mode}/{int(st_time)}/model_{i_episode}.ckpt")

	# # Validation runs every 500 episodes, to select best model
	if i_episode%10==0 and training:
		#set selected VF to 0
		agent.set_active_net(0)
		print("Validating")
		update = i_episode%100==0
		mult = 25 if update else 1

		#Run 50 validation episodes with the current policy
		val_fairness = []
		val_utility = []
		val_mins = []
		for val_eps in range(1*mult):
			print(val_eps, end='\r')
			M_val.reset()
			obs = M_val.get_obs()
			score = 0
			for steps in range(max_steps):
				actions = compute_best_actions(agent, obs, M_val.targets, n_agents, n_resources, M_val.discounted_su, beta=beta, epsilon=0)
				rewards = M_val.step(actions)
				score += sum(rewards)
				obs = M_val.get_obs()
			uti = np.array(M_val.su)/max_steps
			val_fairness.append(np.var(uti)/(np.mean(uti)+0.00001))
			val_utility.append(score/max_steps)
			val_mins.append(min(M_val.su))
		print("Validation Utility: ", np.mean(val_utility))
		print("Validation Fairness: ", np.mean(val_fairness))
		print("Validation Min Utility: ", np.mean(val_mins))

		if logging:
			with summary_writer.as_default():
				tf.summary.scalar("Validation_Utility", float(np.mean(val_utility)), step=i_episode)
				tf.summary.scalar("Validation_Fairness", float(np.mean(val_fairness)), step=i_episode)
				tf.summary.scalar("Validation_Min_Utility", float(np.mean(val_mins)), step=i_episode)

		if  update and np.mean(val_utility)>best_val_utility:
			best_val_utility = np.mean(val_utility)
			#make directory if it doesn't exist
			os.makedirs(f"Models/DoubleDQN/{mode}/{int(st_time)}/best", exist_ok=True)
			agent.save_model(f"Models/DoubleDQN/{mode}/{int(st_time)}/best/best_model.ckpt")
			print("Saved best model")
			#Write the logs to a file
			with open(f"Models/DoubleDQN/{mode}/{int(st_time)}/best/best_log.txt", "w") as f:
				f.write(f"Validation Utility: {np.mean(val_utility)}\n")
				f.write(f"Validation Fairness: {np.mean(val_fairness)}\n")
				f.write(f"Validation Min Utility: {np.mean(val_mins)}\n")
				#Write the episode number, epsilon, simple_obs, reallocate, central_rewards
				f.write(f"Episode: {i_episode}\n")
				f.write(f"Epsilon: {ep_epsilon}\n")
				f.write(f"Simple Obs: {simple_obs}\n")
				f.write(f"Reallocate: {reallocate}\n")
				f.write(f"Central Rewards: {central_rewards}\n")
				f.write(f"Learning Beta: {learning_beta}\n")
				f.write(f"Learning Rate: {learning_rate}\n")
				f_rew = fairness_type
				f.write(f"Fairness Type: {f_rew}\n")
				f.write(f"Warm Start: {warm_start}\n")
				f.write(f"Past Discount: {past_discount}\n")
