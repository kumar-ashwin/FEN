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

from Agents import *
from matching import compute_best_actions
from matthew_envt import get_distance, get_obs, step, MatthewEnvt

#set random seeds for reproducibility
seed_value = 0
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)


# Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

greedy = False
training = True
reallocate = True
central_rewards = True
simple_obs = True
logging = False

st_time = time.time()
if training:
	# Create a summary writer
	mode = "Reallocate" if reallocate else "Fixed"
	mode += "Central" if central_rewards else ""
	log_dir = f"logs/TestOnPolicyVF/{mode}/{int(st_time)}/"
	# log_dir = f"logs/TestOnPolicyVF/{mode}/{int(st_time)}/"
	print("Logging to {} \n\n\n\n".format(log_dir))
	summary_writer = tf.summary.create_file_writer(log_dir)

n_agents=10
n_resources=3
#initialize agents and items. Board size is between 0 and 1
# """
M = MatthewEnvt(n_agents=n_agents, n_resources=n_resources, max_size=0.2, reallocate=reallocate, simple_obs=simple_obs)
obs = M.get_obs()
# """
#initialize agents and items. Board size is between 0 and 1
resource=[]
for i in range(n_resources):
	resource.append(np.random.rand(2))
ant=[]
size=[]
speed=[]
su=[0]*n_agents  #Per agent cumulative rewards
targets=[]
for i in range(n_agents):
	ant.append(np.random.rand(2))
	size.append(0.01+np.random.rand()*0.04)
	speed.append(0.01+size[i])
	targets.append(None) #Target resources, initially None. Stores target resource index, as well as time to reach it
su = np.array(su)
obs = get_obs(ant,resource, targets,size,speed,n_agents)
"""#"""
			
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
n_episode = 20000
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

num_features = len(obs[0][0])

# VF = ValueNetwork(num_features=34, hidden_size=256, learning_rate=0.001)
VF = ValueNetwork(num_features=num_features, hidden_size=256, learning_rate=0.00003)
if not training:
	# VF.load_model("Models_matthew/TestOnPolicyVF_Reallocate/model_15000.ckpt")
	VF.load_model("Models_matthew/TestOnPolicyVF/ReallocateCentral/1688802638.5038037/model_15000.ckpt")

def simple_score(state):
	#For sending to the SimpleValueNetwork class
	#Score is just how far the agents are from the resources.
	if state[-3]==-1:
		return 100
	else:
		return state[-1]

# VF=SimpleValueNetwork(score_func=simple_score, discount_factor=GAMMA)
best_val_utility = 0.0
fairness = []
utility = []
mins = []
while i_episode<n_episode:
	i_episode+=1

	VF_loss = []

	ep_rewards  = [[] for _ in range(n_agents)]
	ep_central_rewards = []
	start_reward = 0.0
	ep_states   = [[] for _ in range(n_agents)]

	score=0  #Central agent score = sum of agent scores
	steps=0
	"""
	M.reset()
	obs = M.get_obs()
	"""
	#initialize agents and items. Board size is between 0 and 1
	resource=[]
	for i in range(n_resources):
		resource.append(np.random.rand(2))
	ant=[]
	size=[]
	speed=[]
	su=[0]*n_agents  #Per agent cumulative rewards
	targets=[]
	for i in range(n_agents):
		ant.append(np.random.rand(2))
		size.append(0.01+np.random.rand()*0.04)
		speed.append(0.01+size[i])
		targets.append(None) #Target resources, initially None. Stores target resource index, as well as time to reach it
	su = np.array(su)

	obs = get_obs(ant,resource, targets,size,speed,n_agents)
	"""#"""

	ep_epsilon = ep_epsilon * epsilon_decay
	if ep_epsilon<min_epsilon:
		ep_epsilon = min_epsilon

	while steps<max_steps:

		steps+=1
		#For each agent, select the action using the central agent

		#Compute post-decision states and add to episode states
		"""
		M.targets  = targets
		M.su = su
		actions = compute_best_actions(VF, obs, M.targets, n_agents, n_resources, M.su, beta=0.0, epsilon=ep_epsilon)
		pd_states = M.get_post_decision_states(obs, actions)
		for i in range(n_agents):
			ep_states[i].append(pd_states[i])
		"""
		actions = compute_best_actions(VF, obs, targets, n_agents, n_resources, su, beta=0.0, epsilon=ep_epsilon)
		#Compute post-decision states and add to episode states
		for i in range(n_agents):
			ant_loc = ant[i]
			s_i = copy.deepcopy(obs[0][i])
			if actions[i]!=-1:
				j = actions[i]
				#apply action
				s_i[-3] = resource[j][0]
				s_i[-2] = resource[j][1]
				s_i[-1] = get_distance(ant_loc,resource[j])/speed[i]
			ep_states[i].append(s_i)
		"""#"""

		#Take a step based on the actions, get rewards and updated agents, resources etc.
		"""
		M.ant = copy.deepcopy(ant)
		M.resource = copy.deepcopy(resource)
		M.size = copy.deepcopy(size)
		M.speed = copy.deepcopy(speed)
		M.targets = copy.deepcopy(targets)
		M.su = copy.deepcopy(su)
		for i in range(n_agents):
			if tuple(M.ant[i]) != tuple(ant[i]):
				print("agents not equal. SANITY CHECK FAILED")
				print(i_episode)
				print("I", i)
				print(M.ant)
				print(ant)
				exit()		
		rewards = M.step(actions)

		# su = M.su
		# ant = M.ant
		# resource = M.resource
		# size = M.size
		# speed = M.speed
		# targets = M.targets
		"""

		
		ant,resource,size,speed,rewards, targets=step(ant,resource, targets, n_resources,n_agents,size,speed,actions, max_size=0.2, reallocate=reallocate)
		su += np.array(rewards)

		# if sum(rewards)>0:
		# 	for i in range(n_agents):
		# 		if tuple(M.ant[i]) != tuple(ant[i]):
		# 			print("agents not equal")
		# 			print("Score", score, sum(rewards), rewards)
		# 			print(steps)
		# 			print("I", i)
		# 			print(M.ant)
		# 			print(ant)
		# 			print("Action", actions)
		# 			print("Targets")
		# 			print(M.targets)
		# 			print(targets)
					# exit()
		"""#"""
		score += sum(rewards)
		ep_central_rewards.append(sum(rewards))

		"""
		obs = M.get_obs()
		su = M.su
		"""
		obs = get_obs(ant,resource, targets, size,speed,n_agents)
		"""#"""
		
		#Calculate episode rewards based on selected policy
		for i in range(n_agents):
			ep_rewards[i].append(rewards[i])
			
		#Update the policies
		if steps%T==0 and training:
			states_all = []
			td_targets_all = []
			for i in range(n_agents):
				ep_rewards[i] = np.array(ep_rewards[i], dtype=np.float_)
				ep_states[i] = np.array(ep_states[i])
				states_all.extend(ep_states[i])

				#Update Value Function for the current policy
				final_state_value = VF.get(ep_states[i])[-1]
				if central_rewards:
					td_targets = discount_rewards(ep_central_rewards, GAMMA, final_state_value=final_state_value)
				else:
					td_targets = discount_rewards(ep_rewards[i], GAMMA, final_state_value=final_state_value)
				td_targets_all.extend(td_targets)
				
			#Shuffle the states and targets
			states_all = np.array(states_all)
			td_targets_all = np.array(td_targets_all)
			indices = np.arange(len(states_all))
			np.random.shuffle(indices)
			states_all = states_all[indices]
			td_targets_all = td_targets_all[indices]
			
			#Update the value function in mini batches
			batch_size = 256
			for i in range(0, len(states_all), batch_size):
				loss = VF.update(states_all[i:i+batch_size], td_targets_all[i:i+batch_size])
				VF_loss.append(loss)
			
			#Reset episode variables
			ep_central_rewards = []
			ep_rewards  = [[] for _ in range(n_agents)]
			ep_states  = [[] for _ in range(n_agents)]
		
		if render:
			M.render()
			time.sleep(0.01)

	print(i_episode)
	print(score/max_steps) #Average reward
	print(su) #Agent rewards
	uti = np.array(su)/max_steps
	print("Fairness",np.var(uti)/np.mean(uti)) #Fairness
	print('epsilon', ep_epsilon)
	if training:
		print("VF learning rate", VF.grad_optimizer._lr)

	fairness.append(np.var(uti)/np.mean(uti))
	utility.append(np.mean(score/max_steps))
	mins.append(min(su))


	print("Average Utility: ", np.mean(utility))
	print("Average Fairness: ", np.mean(fairness))
	print("Average Min Utility: ", np.mean(mins))

	if training:
		with summary_writer.as_default():
			tf.summary.scalar("Value_Loss", float(np.mean(VF_loss)), step=i_episode)
			tf.summary.scalar("Utility", float(score/max_steps), step=i_episode)
			tf.summary.scalar("Fairness", float(np.var(uti)/np.mean(uti)), step=i_episode)
			tf.summary.scalar("Min_Utility", float(min(su)), step=i_episode)

	# Save the model every 500 episodes
	if i_episode%1000==0:
		mode = "Reallocate" if reallocate else "Fixed"
		mode += "Central" if central_rewards else ""
		VF.save_model(f"Models_matthew/TestOnPolicyVF/{mode}/{int(st_time)}/model_{i_episode}.ckpt")

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
			mode = "Reallocate" if reallocate else "Fixed"
			mode += "Central" if central_rewards else ""
			os.makedirs(f"Models_matthew/TestOnPolicyVF/{mode}/{int(st_time)}/best", exist_ok=True)
			VF.save_model(f"Models_matthew/TestOnPolicyVF/{mode}/{int(st_time)}/best/best_model.ckpt")
			print("Saved best model")
			#Write the logs to a file
			with open(f"Models_matthew/TestOnPolicyVF/{mode}/{int(st_time)}/best/best_log.txt", "w") as f:
				f.write(f"Validation Utility: {np.mean(val_utility)}\n")
				f.write(f"Validation Fairness: {np.mean(val_fairness)}\n")
				f.write(f"Validation Min Utility: {np.mean(val_mins)}\n")
				#Write the episode number, epsilon, simple_obs, reallocate, central_rewards
				f.write(f"Episode: {i_episode}\n")
				f.write(f"Epsilon: {ep_epsilon}\n")
				f.write(f"Simple Obs: {simple_obs}\n")
				f.write(f"Reallocate: {reallocate}\n")
				f.write(f"Central Rewards: {central_rewards}\n")

	

				

