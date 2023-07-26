"""
Load the DQN model and run experiments to validate
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

from Agents import SimpleValueNetwork, ValueNetwork, ReplayBuffer
from matching import compute_best_actions, SI_reward
from Environment import MatthewEnvt


# Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

replayBuffer = ReplayBuffer(250000)

greedy = True
training = False
reallocate = False
central_rewards = False
simple_obs = False
logging = False

beta = 0#15
learning_beta = 0
max_size = 0.5

mode = "Reallocate" if reallocate else "Fixed"
mode += "Central" if central_rewards else ""
mode += "" if simple_obs else "Complex"
mode += f"/{learning_beta}"

st_time = time.time()
if logging:
	# Create a summary writer
	log_dir = f"logs/Testing/{mode}/{int(st_time)}/"
	print("Logging to {} \n\n\n\n".format(log_dir))
	summary_writer = tf.summary.create_file_writer(log_dir)

n_agents=10
n_resources=3
#initialize agents and items. Board size is between 0 and 1
M = MatthewEnvt(n_agents=n_agents, n_resources=n_resources, max_size=max_size, reallocate=reallocate, simple_obs=simple_obs)

config = tf.compat.v1.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.compat.v1.Session(config=config)
KTF.set_session(session)
T = 100   # learning frequency for the policy and value networks
totalTime = 0
GAMMA = 0.98
n_episode = 25
max_steps = 500
i_episode = 0
render = False

epsilon = 0.00
min_epsilon = 0.00
epsilon_decay = 0.995
ep_epsilon = epsilon

obs = M.get_obs()
num_features = len(obs[0][0])

model_loc = "Models/DoubleDQN/FixedComplex/0/1689619731/best/best_model.ckpt"
# model_loc = "Models/DoubleDQNRetrain/FixedComplex/0.9/1690240368/model_5000.ckpt"
# model_loc = 'Models/DoubleDQNRetrain/FixedComplex/0.7/1690240427/model_5000.ckpt'
VF = ValueNetwork(num_features=num_features, hidden_size=256, learning_rate=0.000003) 
VF.load_model(model_loc)

def simple_score(state):
	#For sending to the SimpleValueNetwork class
	#Score is just how far the agents are from the resources.
	return 100 if state[-3]==-1 else state[-1]
# VF=SimpleValueNetwork(score_func=simple_score, discount_factor=GAMMA)

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

		#Take a step based on the actions, get rewards and updated agents, resources etc.
		rewards = M.step(actions)
		score += sum(rewards)
		obs = M.get_obs()
					
		if render:
			M.render()
			time.sleep(0.01)

	print(i_episode)
	print(score/max_steps) #Average reward
	print(M.su) #Agent rewards
	uti = np.array(M.su)/max_steps
	print("Fairness",np.var(uti)/np.mean(uti)) #Fairness
	print('epsilon', ep_epsilon)

	fairness.append(np.var(uti)/np.mean(uti))
	utility.append(np.mean(score/max_steps))
	mins.append(min(M.su))


	print("Average Utility: ", np.mean(utility))
	print("Average Fairness: ", np.mean(fairness))
	print("Average Min Utility: ", np.mean(mins))

	if logging:
		with summary_writer.as_default():
			tf.summary.scalar("Utility", float(score/max_steps), step=i_episode)
			tf.summary.scalar("Fairness", float(np.var(uti)/np.mean(uti)), step=i_episode)
			tf.summary.scalar("Min_Utility", float(min(M.su)), step=i_episode)