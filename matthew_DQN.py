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

from matching import get_assignment
from Agents import SimpleValueNetwork, ValueNetwork, ReplayBuffer


replayBuffer = ReplayBuffer(1000000)

# Create a summary writer
log_dir = "logs/metrics_DQN_TD/"
summary_writer = tf.summary.create_file_writer(log_dir)

n_agents=10
n_resources=3
resource=[]
for i in range(n_resources):
	resource.append(np.random.rand(2))
ant=[]
size=[]
speed=[]
targets = []
for i in range(n_agents):
	ant.append(np.random.rand(2)) #Agents
	size.append(0.01+np.random.rand()*0.04) #Agent sizes
	speed.append(0.01+size[i]) #Speeds
	targets.append(None) #Target resources, initially None. Stores target resource index, as well as time to reach it


def get_distance(a,b):
	#Get distance between two points
	return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def get_obs(ant, resource, targets, sizes, speeds, n_agents):
	#Gets the state of the environment (Vector of each agent states)
	#agent positions, resource positions, size vector, speed vector, number of agents
	state=[]
	agents = []
	for i in range(n_agents):
		h=[]
		h.append(ant[i][0])
		h.append(ant[i][1])
		h.append(sizes[i])
		h.append(speeds[i])
		#Current target resource
		if targets[i] is None:
			h.append(-1)
			h.append(-1)
			h.append(0) #Time to reach target resource
		else:
			h.append(resource[targets[i][0]][0]) #Target resource x
			h.append(resource[targets[i][0]][1]) #Target resource y
			h.append(targets[i][1]) #Time to reach target resource
		
		agents.append(h)
	
	state.append(agents)
	state.append(copy.deepcopy(resource))

	return state

def step(ant,resource, targets, n_resources,n_agents,size,speed,action):
	clear_targets = False
	# Actions just decide mapping of agents to resources
	re=[0]*n_agents  #rewards. If an agent picks up a resource, get reward of 1
	for i in range(n_agents):
		if action[i]!=-1:
			targets[i] = [action[i]]
			#Add the time to reach the resource to the targets vector
			targets[i].append(get_distance(ant[i],resource[action[i]])/speed[i])
		
		#Move each agent towards its target resource
		if targets[i] is not None:
			#Other agents can't pick up the resources if they are claimed
			
			#If agent circle will intersect resource circle in next step, pick up resource
			# if targets[i][1]<=1:
			if get_distance(ant[i],resource[targets[i][0]]) - speed[i] <=size[i]:
				ant[i][0] = resource[targets[i][0]][0]
				ant[i][1] = resource[targets[i][0]][1]
				re[i]=1 #Get reward

				#Reset target resource
				resource[targets[i][0]]=np.random.rand(2)
				size[i]=min(size[i]+0.005,0.15)
				# size[i]=min(size[i]+0.05,1.5)
				speed[i]=0.01+size[i]
				targets[i]=None
				clear_targets = True
			else:
				#Move agent towards target resource. Each step, move 1/time_remaining of the way
				ant[i][0]+=(resource[targets[i][0]][0]-ant[i][0])/targets[i][1]
				ant[i][1]+=(resource[targets[i][0]][1]-ant[i][1])/targets[i][1]
				targets[i][1]-=1
		else:
			#Move in a random direction or stay still
			p_move = 0.8
			dr = np.random.rand()*2*np.pi
			if np.random.rand()>p_move:
				ant[i][0]+=np.cos(dr)*speed[i]
				ant[i][1]+=np.sin(dr)*speed[i]
		if ant[i][0]<0:
			ant[i][0]=0
		if ant[i][0]>1:
			ant[i][0]=1
		if ant[i][1]<0:
			ant[i][1]=0
		if ant[i][1]>1:
			ant[i][1]=1

	#If any resources were picked up, reset the targets
	if clear_targets:
		# print("Clearing Targets")
		for i in range(n_agents):
			targets[i]=None


	return ant,resource,size,speed,re

def compute_best_actions(model, obs, targets, n_agents, n_resources, beta=0.0):

	Qvals = [[-1000000 for _ in range(n_resources+1)] for _ in range(n_agents)]
	#First action is to do nothing. This is the default action.
	#Action indexing starts at -1. Shift by 1 to get the correct index
	resource = copy.deepcopy(obs[1])
	for i in range(n_agents):
		h = copy.deepcopy(obs[0][i])
		ant_loc = [h[0],h[1]]

		Qvals[i][0] = float(model.get(np.array([h])))
		occuped_resources = set([targets[j][0] for j in range(n_agents) if targets[j] is not None])

		if targets[i] is None:
			#If the agent can pick another action, get Q values for all actions
			for j in range(n_resources):
				if j not in occuped_resources:
					h = copy.deepcopy(obs[0][i])
					h[-3] = resource[j][0]
					h[-2] = resource[j][1]
					h[-1] = get_distance(ant_loc,resource[j])
					Qvals[i][j+1] = float(model.get(np.array([h])))
				
		#Fairness post processing
		if beta is not 0.0:
			mult = max(0,(su[i] - np.mean(su)))/1000
			# mult = (su[i] - np.mean(su))/1000
			for j in range(len(Qvals[i])):
				if j==0:
					Qvals[i][j] = Qvals[i][j] + beta * mult
				else:
					Qvals[i][j] = Qvals[i][j] - beta * mult
	
	#For each agent, select the action using the central agent given Q values
	actions = get_assignment(Qvals)
	return actions


def simple_score(state):
	#For sending to the SimpleValueNetwork class
	#Score is just how far the agents are from the resources.
	if state[-3]==-1:
		return 0
	else:
		return int(state[-1])
		
			
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
T = 200   # learning frequency for the policy and value networks
totalTime = 0
GAMMA = 0.99
n_episode = 20000
max_steps = 1000
i_episode = 0
render = False

#8 features: x,y,size,speed,target_x,target_y,time_to_target, number of other agents closer to target
#TODO: Add features for relative ratio of size and speed to other agents
VF = ValueNetwork(num_features=7, hidden_size=256, learning_rate=0.001) #TODO: Add num other agents
TargetNetwork = ValueNetwork(num_features=7, hidden_size=256, learning_rate=0.001)
#copy weights from value network to target network
TargetNetwork.set_weights(VF.get_weights())

while i_episode<n_episode:
	i_episode+=1

	#Initializing utilities, average signals, episode states and rewards, meta states and rewards
	avg = [0]*n_agents
	u_bar = [0]*n_agents
	utili = [0]*n_agents
	u = [[] for _ in range(n_agents)]  #Stores historical rewards  for each agent
	max_u = 0.15

	losses = []

	rat = [0.0]*n_agents  #Ratio of performance to average agent

	score=0  #Central agent score = sum of agent scores
	steps=0
	
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

	while steps<max_steps:

		steps+=1
		actions=[]
		#For each agent, select the action using the central agent
		
		actions = compute_best_actions(VF, obs, targets, n_agents, n_resources)

		# print(actions)
		# print(Qvals)
		
		# experience = [obs, actions]
		experience =[copy.deepcopy(obs), copy.deepcopy(actions), copy.deepcopy(targets)]
		# print(len(experience))
		replayBuffer.add(experience)


		#Take a step based on the actions, get rewards and updated agents, resources etc.
		ant,resource,size,speed,rewards=step(ant,resource, targets, n_resources,n_agents,size,speed,actions)
		
		su+=np.array(rewards)
		score += sum(rewards)
		obs = get_obs(ant,resource, targets, size,speed,n_agents)

		#For fairness, capture the average utility of each agent over the history
		for i in range(n_agents):
			u[i].append(rewards[i])
			u_bar[i] = sum(u[i])/len(u[i])

		#Calculate the relative adv/disadv as a ratio for each agent compared to the average agent utility
		# Really similar to SI!!
		for i in range(n_agents):
			avg[i] = sum(u_bar)/len(u_bar)
			if avg[i]!=0:
				rat[i]=(u_bar[i]-avg[i])/avg[i] #How much better or worse you are compared to the average
			else:
				rat[i]=0
			utili[i] = min(1,avg[i]/max_u)  #General utility based on average performance of all agents
			
		#Update: TD Learning
		if steps%T==0:
			#Update the value function
			if len(replayBuffer.buffer) < 100000/n_agents:
				continue
			
			print("Updating Value Function")
			#Sample a batch of experiences from the replay buffer
			experiences = replayBuffer.sample(int(10))

			for experience in experiences:
				#Compute best action
				loc_obs, loc_actions, loc_targets = experience
				opt_actions = compute_best_actions(VF, loc_obs, loc_targets, n_agents, n_resources)
				new_ant,new_resource,new_size,new_speed,new_rewards=step(ant,resource, targets, n_resources,n_agents,size,speed,opt_actions)
				new_obs = get_obs(new_ant,new_resource, targets, new_size,new_speed,n_agents)

				#Perform batched updates
				states = np.array([loc_obs[0][i] for i in range(n_agents)])
				target_values = np.array([new_rewards[i] + GAMMA * TargetNetwork.get(np.array([new_obs[0][i]])) for i in range(n_agents)])
				target_values = target_values.reshape(-1)
				
				loss = VF.update(states, target_values)
				losses.append(loss)
		
		if render:
			for i in range(n_agents):
				theta = np.arange(0, 2*np.pi, 0.01)
				x = ant[i][0] + size[i] * np.cos(theta)
				y = ant[i][1] + size[i] * np.sin(theta)
				plt.plot(x, y)
				if targets[i] is not None:
					#plot a line from ant to target
					plt.plot([ant[i][0],resource[targets[i][0]][0]],[ant[i][1],resource[targets[i][0]][1]], color = 'red')
			for i in range(n_resources):
				plt.scatter(resource[i][0], resource[i][1], color = 'green')
			plt.axis("off")
			plt.axis("equal")
			plt.xlim(0 , 1)
			plt.ylim(0 , 1)
			plt.ion()
			plt.pause(0.1)
			plt.close()

	print(i_episode)
	print(score/max_steps) #Average reward
	print(su) #Agent rewards
	uti = np.array(su)/max_steps
	print("Fairness",np.var(uti)/np.mean(uti)) #Fairness
	
	with summary_writer.as_default():
		if len(losses):
			tf.summary.scalar("Value_Loss", float(np.mean(losses)), step=i_episode)
		tf.summary.scalar("Utility", float(score/max_steps), step=i_episode)
		tf.summary.scalar("Fairness", float(np.var(uti)/np.mean(uti)), step=i_episode)
		tf.summary.scalar("Min_Utility", float(min(su)), step=i_episode)

	#update the target network every 50 episodes
	if i_episode%10==0:
		TargetNetwork.set_weights(VF.get_weights())

	# Save the model every 500 episodes
	if i_episode%1000==0:
		VF.save_model(f"Models_matthew/DQN_VF/model_{i_episode}.ckpt")