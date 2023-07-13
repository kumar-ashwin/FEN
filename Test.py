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
# from matthew_envt import get_distance, get_obs, step, MatthewEnvt
from matthew_mod_envt import MatthewEnvt


# Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


training = False
reallocate = False
simple_obs = True
logging = False

st_time = time.time()
if logging:
	# Create a summary writer
	mode = "Reallocate" if reallocate else "Fixed"
	mode += "Simple" if simple_obs else "Complex"
	log_dir = f"logs/Test/{mode}/{int(st_time)}/"
	print("Logging to {} \n\n\n\n".format(log_dir))
	summary_writer = tf.summary.create_file_writer(log_dir)

def simple_score(state):
	#For sending to the SimpleValueNetwork class
	#Score is just how far the agents are from the resources.
	if state[-3]==-1:
		return 100
	else:
		# return state[-1]/state[4]
		return state[-1]
def get_CofVar(utilities):
	#Function to calculate the Coefficient of Variation
	mean = np.mean(utilities)
	std = np.std(utilities)
	return std/mean

n_agents=10
n_resources=3
#initialize agents and items. Board size is between 0 and 1
M = MatthewEnvt(n_agents=n_agents, n_resources=n_resources, max_size=0.5, min_size = 0.01, size_update=0.005, base_speed=0.01, reallocate=reallocate, simple_obs=simple_obs)

config = tf.compat.v1.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.compat.v1.Session(config=config)
KTF.set_session(session)

n_episode = 10
max_steps = 500
i_episode = 0
render = False

epsilon = ep_epsilon = 0.0

obs = M.get_obs()
num_features = len(obs[0][0])

VF = ValueNetwork(num_features=num_features, hidden_size=256, learning_rate=0.001)
VF.load_model("Models_matthew/DQN/Reallocate/1688928248/best/best_model.ckpt")
VF=SimpleValueNetwork(score_func=simple_score)

betas= [0]
betas= [0.0,0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
# betas= [0.0, 1, 10, 100]
utilities = []
fairness = []
social_welfare = []
CofVar = []
min_utilities = []
max_utilities = []
mins = []
for beta in betas:
	i_episode = 0
	iters = []
	results_fairness = []
	results_utility = []
	social_welfare_results = []
	CofVar_results = []
	min_utility_results = []
	max_utility_results = []
	while i_episode<n_episode:
		i_episode+=1

		VF_loss = []

		score=0  #Central agent score = sum of agent scores
		steps=0
		
		M.reset()
		obs = M.get_obs()


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

		uti = np.array(M.su)/max_steps

		iters.append(i_episode)
		#Save the results
		results_fairness.append(np.var(uti)/np.mean(uti))
		results_utility.append(score/max_steps)
		social_welfare_results.append(score)
		CofVar_results.append(get_CofVar(M.su))
		min_utility_results.append(min(M.su))
		max_utility_results.append(max(M.su))

		print(i_episode)
		print('beta', beta)
		print(score/max_steps) #Average reward
		print(M.su) #Agent rewards
		print(M.size)
		print(M.speed)
		print(M.agent_types)
		print('Avg Utility', np.mean(results_utility))
		print('Avg Fairness', np.mean(results_fairness))
		print('Avg Min Utility', np.mean(min_utility_results))
	avg_utility = np.mean(results_utility)
	avg_fairness = np.mean(results_fairness)
	avg_social_welfare = np.mean(social_welfare_results)
	avg_CofVar = np.mean(CofVar_results)
	avg_min_utility = np.mean(min_utility_results)
	avg_max_utility = np.mean(max_utility_results)

	print("Beta: ", beta)
	print("Average utility: ", avg_utility)
	print("Average fairness: ", avg_fairness)
	print("Average social welfare: ", avg_social_welfare)
	print("Average CofVar: ", avg_CofVar)
	print("Average min utility: ", avg_min_utility)
	print("Average max utility: ", avg_max_utility)
	print()

	utilities.append(avg_utility)
	fairness.append(avg_fairness)
	social_welfare.append(avg_social_welfare)
	CofVar.append(avg_CofVar)
	min_utilities.append(avg_min_utility)
	max_utilities.append(avg_max_utility)

	# #plotting
	plt.plot(utilities, fairness, color = 'green', marker = 'o')
	# plt.ion()
	# plt.pause(0.4)
	# plt.close()
#Save the results
plt.xlabel('Average Utility')
plt.ylabel('Fairness')
plt.title('Fairness vs Utility')
plt.savefig('Fairness_vs_Utility.png')

