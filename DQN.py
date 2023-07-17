"""
TRAP version of the environment.
Casting it as a resource allocation problem, each agent gives preference over the resources, and a central decision maker allocates the resources to the agents.
Instead of simulating motion step by step, calculate the time needed to reach the reource, and keep the agent out of the simulation until that time.
Size of agent is proportional to speed, eating food increases size and speed.
Comments and explanations added by Ashwin Kumar (Washington University in St Louis)

Based on a DQN based approach instead of a policy gradient based approach.
"""
import os, sys, time  
import numpy as np
import tensorflow as tf
import random
from keras.utils import np_utils,to_categorical
# import keras.backend.tensorflow_backend as KTF
from tensorflow.python.keras import backend as KTF
from keras import backend as K
import copy
import matplotlib.pyplot as plt

from tensorboard.plugins.hparams import api as hp

from Agents import SimpleValueNetwork, ValueNetwork, ReplayBuffer
from matching import compute_best_actions, SI_reward
from matthew_envt import get_distance, get_obs, step, MatthewEnvt


# Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

replayBuffer = ReplayBuffer(1000000)

greedy = False
training = True
reallocate = True
central_rewards = False
simple_obs = True
logging = False

beta = 0
learning_beta = 0

if central_rewards:
	print("Central rewards for DQN not implemented yet. Exiting")
	exit()

mode = "Reallocate" if reallocate else "Fixed"
mode += "Central" if central_rewards else ""
mode += "Simple" if simple_obs else "Complex"
mode += "_penalty"
mode += "_SqSpeed"
mode += f"/{learning_beta}"

st_time = time.time()
if training:
	# Create a summary writer
	log_dir = f"logs/DQN/{mode}/{int(st_time)}/"
	print("Logging to {} \n\n\n\n".format(log_dir))
	summary_writer = tf.summary.create_file_writer(log_dir)

n_agents=10
n_resources=3
#initialize agents and items. Board size is between 0 and 1
M = MatthewEnvt(n_agents=n_agents, n_resources=n_resources, max_size=0.5, reallocate=reallocate, simple_obs=simple_obs)
			
def discount_rewards(rewards,gamma, final_state_value=0.0):
		running_total = final_state_value
		discounted = np.zeros_like(rewards)
		
		for r in reversed(range(len(rewards))):
			running_total = running_total *gamma + rewards[r]
			discounted[r] = running_total
		return discounted

config = tf.compat.v1.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.compat.v1.Session(config=config)
KTF.set_session(session)
T = 100   # learning frequency for the policy and value networks
totalTime = 0
GAMMA = 0.98
n_episode = 5000
max_steps = 1000
i_episode = 0
render = False

epsilon = 0.5
min_epsilon = 0.05
if greedy:
	epsilon = 0.00
	min_epsilon = 0.00
epsilon_decay = 0.99
ep_epsilon = epsilon

obs = M.get_obs()
num_features = len(obs[0][0])

VF = ValueNetwork(num_features=num_features, hidden_size=256, learning_rate=0.001) #TODO: Add num other agents
TargetNetwork = ValueNetwork(num_features=num_features, hidden_size=256, learning_rate=0.001)
TargetNetwork.set_weights(VF.get_weights())

if not training:
	VF.load_model("Models_matthew/DQN/Reallocate/1688928248/best/best_model.ckpt")

def simple_score(state):
	#For sending to the SimpleValueNetwork class
	#Score is just how far the agents are from the resources.
	if state[-3]==-1:
		return 100
	else:
		return state[-1]

VF=SimpleValueNetwork(score_func=simple_score, discount_factor=GAMMA)
best_val_utility = 0.0
fairness = []
utility = []
mins = []
while i_episode<n_episode:
	i_episode+=1

	VF_loss = []

	score=0  #Central agent score = sum of agent scores
	steps=0
	
	M.reset()
	obs = M.get_obs()

	ep_epsilon = ep_epsilon * epsilon_decay
	if ep_epsilon<min_epsilon:
		ep_epsilon = min_epsilon

	while steps<max_steps:

		steps+=1
		#For each agent, select the action using the central agent
		actions = compute_best_actions(VF, obs, M.targets, n_agents, n_resources, M.su, beta=beta, epsilon=ep_epsilon)
		
		#Add to replay buffer
		experience =[copy.deepcopy(M.get_state()), copy.deepcopy(actions)]
		replayBuffer.add(experience)

		#Take a step based on the actions, get rewards and updated agents, resources etc.
		rewards = M.step(actions)
		score += sum(rewards)
		obs = M.get_obs()
					
		#Update the policies
		if steps%T==0 and training:
			#Update the value function
			if len(replayBuffer.buffer) < 100000/n_agents:
				continue
			
			# print("Updating Value Function")
			M_train = MatthewEnvt(n_agents=n_agents, n_resources=n_resources, max_size=0.2, reallocate=reallocate, simple_obs=simple_obs)
			#Sample a batch of experiences from the replay buffer
			experiences = replayBuffer.sample(int(10))

			for experience in experiences:
				M_train.set_state(experience[0])
				#Compute best action
				_, loc_actions = experience
				loc_obs = M_train.get_obs()
				opt_actions = compute_best_actions(VF, obs, M_train.targets, n_agents, n_resources, M_train.su, beta=0.0, epsilon=0)
				new_rewards = M_train.step(opt_actions)
				if central_rewards:
					new_rewards = [np.mean(new_rewards)]*n_agents

				if learning_beta>0:
					fair_rewards = SI_reward(M_train.su, direction="adv")
					print(fair_rewards)
					new_rewards = new_rewards + learning_beta*fair_rewards

				new_obs = M_train.get_obs()

				#Perform batched updates
				states = np.array([loc_obs[0][i] for i in range(n_agents)])
				target_values = np.array([new_rewards[i] + GAMMA * TargetNetwork.get(np.array([new_obs[0][i]])) for i in range(n_agents)])
				target_values = target_values.reshape(-1)
				
				loss = VF.update(states, target_values)
				VF_loss.append(loss)
		
		if render:
			M.render()
			time.sleep(0.01)

	print(i_episode)
	print(score/max_steps) #Average reward
	print(M.su) #Agent rewards
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

	if training:
		with summary_writer.as_default():
			if len(VF_loss)>0:
				tf.summary.scalar("Value_Loss", float(np.mean(VF_loss)), step=i_episode)
			tf.summary.scalar("Utility", float(score/max_steps), step=i_episode)
			tf.summary.scalar("Fairness", float(np.var(uti)/np.mean(uti)), step=i_episode)
			tf.summary.scalar("Min_Utility", float(min(M.su)), step=i_episode)
		
		#update the target network every 10 episodes
		if i_episode%10==0:
			TargetNetwork.set_weights(VF.get_weights())

	# Save the model every 500 episodes
	if i_episode%1000==0:
		VF.save_model(f"Models_matthew/DQN/{mode}/{int(st_time)}/model_{i_episode}.ckpt")

	# # Validation runs every 500 episodes, to select best model
	if i_episode%200==0 and training:
		print("Validating")
		
		#Run 50 validation episodes with the current policy
		M_val = MatthewEnvt(n_agents=n_agents, n_resources=n_resources, max_size=0.2, reallocate=reallocate, simple_obs=True)
		val_fairness = []
		val_utility = []
		val_mins = []
		for val_eps in range(50):
			print(val_eps, end='\r')
			M_val.reset()
			obs = M_val.get_obs()
			score = 0
			for steps in range(max_steps):
				actions = compute_best_actions(VF, obs, M.targets, n_agents, n_resources, M.su, beta=0.0, epsilon=0)
				rewards = M_val.step(actions)
				score += sum(rewards)
				obs = M_val.get_obs()
			uti = np.array(M_val.su)/max_steps
			val_fairness.append(np.var(uti)/np.mean(uti))
			val_utility.append(score/max_steps)
			val_mins.append(min(M_val.su))
		print("Validation Utility: ", np.mean(val_utility))
		print("Validation Fairness: ", np.mean(val_fairness))
		print("Validation Min Utility: ", np.mean(val_mins))

		with summary_writer.as_default():
			tf.summary.scalar("Validation_Utility", float(np.mean(val_utility)), step=i_episode)
			tf.summary.scalar("Validation_Fairness", float(np.mean(val_fairness)), step=i_episode)
			tf.summary.scalar("Validation_Min_Utility", float(np.mean(val_mins)), step=i_episode)

		if np.mean(val_utility)>best_val_utility:
			best_val_utility = np.mean(val_utility)
			#make directory if it doesn't exist
			os.makedirs(f"Models_matthew/DQN/{mode}/{int(st_time)}/best", exist_ok=True)
			VF.save_model(f"Models_matthew/DQN/{mode}/{int(st_time)}/best/best_model.ckpt")
			print("Saved best model")
			#Write the logs to a file
			with open(f"Models_matthew/DQN/{mode}/{int(st_time)}/best/best_log.txt", "w") as f:
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

	

				
