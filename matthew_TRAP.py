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

# Create a summary writer
log_dir = "logs/metrics_DQN"
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


import gurobipy as gp
from gurobipy import GRB

def get_assignment(Qvalues):
	n_agents = len(Qvalues)
	n_resources = len(Qvalues[0])
	#Create a model
	m = gp.Model("mip1")
	m.setParam('OutputFlag', 0)
	#Create variables
	x = m.addVars(n_agents, n_resources, vtype=GRB.BINARY, name="x")
	#Set objective
	m.setObjective(sum(sum(Qvalues[i][j]*x[i,j] for j in range(n_resources)) for i in range(n_agents)), GRB.MAXIMIZE)
	#Add constraints
	m.addConstrs(sum(x[i,j] for j in range(n_resources))==1 for i in range(n_agents)) # Each agent can only be assigned to exactly one resource
	m.addConstrs(sum(x[i,j] for i in range(n_agents))<=1 for j in range(1,n_resources)) # Each resource except the first one can only be assigned to one agent
	#Solve
	m.optimize()
	#Get solution
	assignment = []
	for i in range(n_agents):
		for j in range(n_resources):
			if x[i,j].x==1:
				assignment.append(j-1)
	return assignment


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
			if targets[i][1]<=1:
				ant[i][0] = resource[targets[i][0]][0]
				ant[i][1] = resource[targets[i][0]][1]
				re[i]=1 #Get reward

				#Reset target resource
				resource[targets[i][0]]=np.random.rand(2)
				size[i]=min(size[i]+0.005,0.25)
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

	#If any resources were picked up, reset the targets
	if clear_targets:
		print("Clearing Targets")
		for i in range(n_agents):
			targets[i]=None

	return ant,resource,size,speed,re

class ValueNetwork():
	def __init__(self, num_features, hidden_size, learning_rate=.01):
		self.num_features = num_features
		self.hidden_size = hidden_size
		self.tf_graph = tf.Graph()
		with self.tf_graph.as_default():
			self.session = tf.compat.v1.Session()


			self.observations = tf.compat.v1.placeholder(shape=[None, self.num_features], dtype=tf.float32)
			self.W = [
				tf.compat.v1.get_variable("W1", shape=[self.num_features, self.hidden_size]),
				tf.compat.v1.get_variable("W2", shape=[self.hidden_size, self.hidden_size]),
				tf.compat.v1.get_variable("W3", shape=[self.hidden_size, 1])
			]
			self.layer_1 = tf.nn.relu(tf.matmul(self.observations, self.W[0]))
			self.layer_2 = tf.nn.relu(tf.matmul(self.layer_1, self.W[1]))
			self.output = tf.reshape(tf.matmul(self.layer_2, self.W[2]), [-1])

			self.rollout = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
			self.loss = tf.compat.v1.losses.mean_squared_error(self.output, self.rollout)
			self.grad_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
			self.minimize = self.grad_optimizer.minimize(self.loss)

			self.saver = tf.compat.v1.train.Saver(self.W,max_to_keep=3000)

			init = tf.compat.v1.global_variables_initializer()
			self.session.run(init)

	def get(self, states):
		value = self.session.run(self.output, feed_dict={self.observations: states})
		return value

	def update(self, states, discounted_rewards):
		_, loss = self.session.run([self.minimize, self.loss], feed_dict={
			self.observations: states, self.rollout: discounted_rewards
		})
		return loss
		
	def save_model(self, save_path):
		with self.tf_graph.as_default():
			self.saver.save(self.session, save_path)
	
	def load_model(self, save_path):
		with self.tf_graph.as_default():
			self.saver.restore(self.session, save_path)

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
T = 50   # learning frequency for the policy and value networks
totalTime = 0
GAMMA = 0.99
n_episode = 20000
max_steps = 1000
i_episode = 0
n_actions = 5
n_signal = 1  #Number of policies to have
render = False

# Pi = [] #Policy network
# V = [] #Value function (Baseline)
#Make one pair of networks for each type of policy (I presume first one will be the greedy one)
VF = ValueNetwork(num_features=7, hidden_size=256, learning_rate=0.001)

while i_episode<n_episode:
	i_episode+=1

	#Initializing utilities, average signals, episode states and rewards, meta states and rewards
	avg = [0]*n_agents
	u_bar = [0]*n_agents
	utili = [0]*n_agents
	u = [[] for _ in range(n_agents)]  #Stores historical rewards  for each agent
	max_u = 0.15

	VF_loss = []

	ep_actions  = [[] for _ in range(n_agents)]
	ep_rewards  = [[] for _ in range(n_agents)]
	ep_states   = [[] for _ in range(n_agents)]

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
		Qvals = [[-1000000 for _ in range(n_resources+1)] for _ in range(n_agents)]
		#First action is to do nothing. This is the default action.
		#Action indexing starts at -1. Shift by 1 to get the correct index
		for i in range(n_agents):
			h = copy.deepcopy(obs[0][i])
			ep_states[i].append(h)

			Qvals[i][0] = float(VF.get(np.array([h])))
			if targets[i] is None:
				#If the agent can pick another action, get Q values for all actions
				for j in range(n_resources):
					h = copy.deepcopy(obs[0][i])
					h[-3] = resource[j][0]
					h[-2] = resource[j][1]
					h[-1] = get_distance(ant[i],resource[j])
					Qvals[i][j+1] = float(VF.get(np.array([h])))
		
		#For each agent, select the action using the central agent given Q values
		actions = get_assignment(Qvals)
		
		# print(actions)
		# print(Qvals)
		
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

		#Calculate episode rewards based on selected policy
		for i in range(n_agents):
			ep_rewards[i].append(rewards[i])
			
		#Update the policies
		if steps%T==0:
			for i in range(n_agents):
				ep_actions[i] = np.array(ep_actions[i])
				ep_rewards[i] = np.array(ep_rewards[i], dtype=np.float_)
				ep_states[i] = np.array(ep_states[i])

				#Update Value Function for the current policy
				final_state_value = VF.get(ep_states[i])[-1]
				# print("Final_state_value", final_state_value)
				td_targets = discount_rewards(ep_rewards[i],GAMMA, final_state_value=final_state_value)
				v_loss = VF.update(ep_states[i], td_targets)
				VF_loss.append(v_loss)

			
			#Reset episode variables
			ep_actions  = [[] for _ in range(n_agents)]
			ep_rewards  = [[] for _ in range(n_agents)]
			ep_states  = [[] for _ in range(n_agents)]
		
		if render:
			for i in range(n_agents):
				theta = np.arange(0, 2*np.pi, 0.01)
				x = ant[i][0] + size[i] * np.cos(theta)
				y = ant[i][1] + size[i] * np.sin(theta)
				plt.plot(x, y)
			for i in range(n_resources):
				plt.scatter(resource[i][0], resource[i][1], color = 'green')
			plt.axis("off")
			plt.axis("equal")
			plt.xlim(0 , 1)
			plt.ylim(0 , 1)
			plt.ion()
			plt.pause(0.4)
			plt.close()

	print(i_episode)
	print(score/max_steps) #Average reward
	print(su) #Agent rewards
	uti = np.array(su)/max_steps
	print("Fairness",np.var(uti)/np.mean(uti)) #Fairness
	
	with summary_writer.as_default():
		tf.summary.scalar("Value_Loss", float(np.mean(VF_loss[0])), step=i_episode)
		tf.summary.scalar("Utility", float(score/max_steps), step=i_episode)
		tf.summary.scalar("Fairness", float(np.var(uti)/np.mean(uti)), step=i_episode)

	# Save the model every 500 episodes
	if i_episode%1000==0:
		VF.save_model(f"Models_matthew/OnPolicyVF/model_{i_episode}.ckpt")