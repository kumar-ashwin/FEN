"""
Code from FEN (Jiechuan Jiang and Zongqing Lu)
Comments and explanations added by Ashwin Kumar (Washington University in St Louis)
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

n_agent=10
n_resource=3
resource=[]
for i in range(n_resource):
	resource.append(np.random.rand(2))
ant=[]
size=[]
speed=[]
for i in range(n_agent):
	ant.append(np.random.rand(2)) #Agents
	size.append(0.01+np.random.rand()*0.04) #Agent sizes
	speed.append(0.01+size[i]) #Speeds

def get_obs(ant,resource,si,sp,n_agent):
	#Gets the state of the environment (Vector of each agent states)
	#agent positions, resource positions, size vector, speed vector, number of agents
	state=[]
	for i in range(n_agent):
		h=[]
		h.append(ant[i][0])
		h.append(ant[i][1])
		h.append(si[i])
		h.append(sp[i])
		j=0
		mi = 10
		for k in range(len(resource)):
			#for each resource, find the distance (euclidean). Save location of closest resource.
			t = (resource[k][0]-ant[i][0])**2+(resource[k][1]-ant[i][1])**2
			if t<mi:
				j = k
				mi = t
		h.append(resource[j][0])
		h.append(resource[j][1])
		state.append(h)
	return state

def step(ant,resource,n_resource,n_agent,size,speed,action):
	re=[0]*n_agent  #rewards. If an agent picks up a resource, get reward of 1
	for i in range(n_agent):
		#Each agent gets 5 actions. Move up, down, left, right, nothing (0 to 5)
		if action[i]==1:
			ant[i][0]-=speed[i]
			if ant[i][0]<0:
				ant[i][0]=0
		if action[i]==2:
			ant[i][0]+=speed[i]
			if ant[i][0]>1:
				ant[i][0]=1
		if action[i]==3:
			ant[i][1]-=speed[i]
			if ant[i][1]<0:
				ant[i][1]=0
		if action[i]==4:
			ant[i][1]+=speed[i]
			if ant[i][1]>1:
				ant[i][1]=1
	for i in range(n_resource):
		for j in range(n_agent):
			#Trivial ordering. First agent that overlaps with the resource gets it
			#Increase size (and also speed) of said agent
			# Also generate a new resource to replace it
			if (resource[i][0]-ant[j][0])**2+(resource[i][1]-ant[j][1])**2<size[j]**2:
				re[j]=1
				resource[i]=np.random.rand(2)
				size[j]=min(size[j]+0.005,0.15)
				speed[j]=0.01+size[j]
				break

	return ant,resource,size,speed,re

class PPOPolicyNetwork():
	def __init__(self, num_features, layer_size, num_actions, epsilon=.2,
				 learning_rate=9e-4):
		self.tf_graph = tf.Graph()

		with self.tf_graph.as_default():
			self.session = tf.compat.v1.Session()

			self.observations = tf.compat.v1.placeholder(shape=[None, num_features], dtype=tf.float32)
			self.W = [
				tf.compat.v1.get_variable("W1", shape=[num_features, layer_size]),
				tf.compat.v1.get_variable("W2", shape=[layer_size, layer_size]),
				tf.compat.v1.get_variable("W3", shape=[layer_size, num_actions])
			]

			self.saver = tf.compat.v1.train.Saver(self.W,max_to_keep=3000)
			
			self.output = tf.nn.relu(tf.matmul(self.observations, self.W[0]))
			self.output = tf.nn.relu(tf.matmul(self.output, self.W[1]))
			self.output = tf.nn.softmax(tf.matmul(self.output, self.W[2]))

			self.advantages = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

			self.chosen_actions = tf.compat.v1.placeholder(shape=[None, num_actions], dtype=tf.float32)
			self.old_probabilities = tf.compat.v1.placeholder(shape=[None, num_actions], dtype=tf.float32)

			self.new_responsible_outputs = tf.reduce_sum(input_tensor=self.chosen_actions*self.output, axis=1)
			self.old_responsible_outputs = tf.reduce_sum(input_tensor=self.chosen_actions*self.old_probabilities, axis=1)

			self.ratio = self.new_responsible_outputs/self.old_responsible_outputs

			self.loss = tf.reshape(
							tf.minimum(
								tf.multiply(self.ratio, self.advantages), 
								tf.multiply(tf.clip_by_value(self.ratio, 1-epsilon, 1+epsilon), self.advantages)),
							[-1]
						) - 0.03*self.new_responsible_outputs*tf.math.log(self.new_responsible_outputs + 1e-10)
			self.loss = -tf.reduce_mean(input_tensor=self.loss)

			self.W0_grad = tf.compat.v1.placeholder(dtype=tf.float32)
			self.W1_grad = tf.compat.v1.placeholder(dtype=tf.float32)
			self.W2_grad = tf.compat.v1.placeholder(dtype=tf.float32)

			self.gradient_placeholders = [self.W0_grad, self.W1_grad, self.W2_grad]
			self.trainable_vars = self.W
			self.gradients = [(np.zeros(var.get_shape()), var) for var in self.trainable_vars]

			self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
			self.get_grad = self.optimizer.compute_gradients(self.loss, self.trainable_vars)
			self.apply_grad = self.optimizer.apply_gradients(zip(self.gradient_placeholders, self.trainable_vars))
			init = tf.compat.v1.global_variables_initializer()
			self.session.run(init)

	def get_dist(self, states):
		#Distribution of different actions
		dist = self.session.run(self.output, feed_dict={self.observations: states})
		return dist

	def update(self, states, chosen_actions, ep_advantages):
		old_probabilities = self.session.run(self.output, feed_dict={self.observations: states})
		self.session.run(self.apply_grad, feed_dict={
			self.W0_grad: self.gradients[0][0],
			self.W1_grad: self.gradients[1][0],
			self.W2_grad: self.gradients[2][0],

		})
		self.gradients, loss = self.session.run([self.get_grad, self.output], feed_dict={
			self.observations: states,
			self.advantages: ep_advantages,
			self.chosen_actions: chosen_actions,
			self.old_probabilities: old_probabilities
		})
		return loss 
		
	def save_w(self,name):
		self.saver.save(self.session,name+'.ckpt')
	def restore_w(self,name):
		self.saver.restore(self.session,name+'.ckpt')
	
	def save_model(self, save_path):
		with self.tf_graph.as_default():
			self.saver.save(self.session, save_path)
	
	def load_model(self, save_path):
		with self.tf_graph.as_default():
			self.saver.restore(self.session, save_path)
			
config = tf.compat.v1.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.compat.v1.Session(config=config)
KTF.set_session(session)
T = 50   # learning frequency for the policy and value networks
totalTime = 0
GAMMA = 0.98
n_episode = 20
max_steps = 1000
i_episode = 0
n_actions = 5
n_signal = 1  #Number of policies to have
render = False

Pi = PPOPolicyNetwork(num_features=6, num_actions=n_actions,layer_size=256,epsilon=0.2,learning_rate=0.0003)
Pi.load_model("Models_matthew/PPO_Pi_2/model_11000.ckpt")


# betas= [0.0,0.1,0.2,0.5,1.0]#, 2, 5, 10]
# betas= [0.0,0.1,0.2,0.4, 0.6, 0.8, 1.0, 2, 4, 6, 8, 10]
betas= [4]
utilities = []
fairness = []
social_welfare = []
CofVar = []
min_utilities = []
max_utilities = []

def get_CofVar(utilities):
	#Function to calculate the Coefficient of Variation
	mean = np.mean(utilities)
	std = np.std(utilities)
	return std/mean

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

		#Initializing utilities, average signals, episode states and rewards, meta states and rewards
		avg = [0]*n_agent
		u_bar = [0]*n_agent
		utili = [0]*n_agent
		u = [[] for _ in range(n_agent)]  #Stores historical rewards  for each agent
		max_u = 0.15

		ep_actions  = [[] for _ in range(n_agent)]
		ep_rewards  = [[] for _ in range(n_agent)]
		ep_states   = [[] for _ in range(n_agent)]

		rat = [0.0]*n_agent  #Ratio of performance to average agent

		score=0  #Central agent score = sum of agent scores
		steps=0
		
		#initialize agents and items. Board size is between 0 and 1
		resource=[]
		for i in range(n_resource):
			resource.append(np.random.rand(2))
		ant=[]
		size=[]
		speed=[]
		su=[0]*n_agent  #Per agent cumulative rewards
		for i in range(n_agent):
			ant.append(np.random.rand(2))
			size.append(0.01+np.random.rand()*0.04)
			speed.append(0.01+size[i])
		su = np.array(su)

		obs = get_obs(ant,resource,size,speed,n_agent)

		while steps<max_steps:

			steps+=1
			action=[]
			#For each agent, select the action as per the probability distribution given by the PPOPolicy 
			# corresponding to the selected Meta policy for agent i -> Pi
			for i in range(n_agent):
				h = copy.deepcopy(obs[i])
				p = Pi.get_dist(np.array([h]))[0]
				
				#Modify probability to make it fairer
				#TODO Question: What is a reasonable way to skew probabilities with SI? 
				# An exponential function may be better, but going with a linear function for now
				# beta = 0.5
				p = np.array(p)
				pbar = sum(p)/len(p)
				
				# mult = rat[i]
				#Use the total reward
				mult = max(0,(su[i] - np.mean(su)))/100

				# p = p + beta * (pbar - p) * mult
				p = p + beta * (pbar - p) * mult

				#Better strategy: Don't go for the resource that worse off people want



				#Normalize
				p = p - min(p) + 0.01
				p = p/sum(p) 

				action.append(np.random.choice(range(n_actions), p=p))
			
			#Take a step based on the actions, get rewards and updated agents, resources etc.
			ant,resource,size,speed,rewards=step(ant,resource,n_resource,n_agent,size,speed,action)
			
			su+=np.array(rewards)
			score += sum(rewards)
			obs = get_obs(ant,resource,size,speed,n_agent)

			#For fairness, capture the average utility of each agent over the history
			for i in range(n_agent):
				u[i].append(rewards[i])
				u_bar[i] = sum(u[i])/len(u[i])

			#Calculate the relative adv/disadv as a ratio for each agent compared to the average agent utility
			# Really similar to SI!!
			for i in range(n_agent):
				avg[i] = sum(u_bar)/len(u_bar)
				if avg[i]!=0:
					rat[i]=(u_bar[i]-avg[i])/avg[i] #How much better or worse you are compared to the average
				else:
					rat[i]=0
				utili[i] = min(1,avg[i]/max_u)  #General utility based on average performance of all agents

			if render:
				for i in range(n_agent):
					theta = np.arange(0, 2*np.pi, 0.01)
					x = ant[i][0] + size[i] * np.cos(theta)
					y = ant[i][1] + size[i] * np.sin(theta)
					plt.plot(x, y)
				for i in range(n_resource):
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
		
		iters.append(i_episode)
		#Save the results
		results_fairness.append(np.var(uti)/np.mean(uti))
		results_utility.append(score/max_steps)
		social_welfare_results.append(score)
		CofVar_results.append(get_CofVar(su))
		min_utility_results.append(min(su))
		max_utility_results.append(max(su))
		
		# #plotting
		# plt.plot(iters, results_utility, color = 'green')
		# plt.plot(iters, results_fairness, color = 'blue')
		# # plt.axis("off")
		# # plt.axis("equal")
		# plt.xlim(0 , n_episode)
		# # plt.ylim(0 , 1)
		# plt.ion()
		# plt.pause(0.4)
		# plt.close()
	
	avg_utility = sum(results_utility)/len(results_utility)
	avg_fairness = sum(results_fairness)/len(results_fairness)
	avg_social_welfare = sum(social_welfare_results)/len(social_welfare_results)
	avg_CofVar = sum(CofVar_results)/len(CofVar_results)
	avg_min_utility = sum(min_utility_results)/len(min_utility_results)
	avg_max_utility = sum(max_utility_results)/len(max_utility_results)

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
	plt.ion()
	plt.pause(0.4)
	plt.close()

# #Save the plot
# plt.plot(utilities, fairness, color = 'green', marker = 'o')
# #Add text to the markers for beta
# for i in range(len(utilities)):
# 	plt.text(utilities[i], fairness[i], str(betas[i]))
# plt.title("Tradeoff between fairness and average utility")
# plt.xlabel("Average Utility")
# plt.ylabel("Fairness")
# plt.savefig("matthew_tradeoff.png")
# plt.close()

# #Plot social welfare vs min utility
# plt.plot(social_welfare, min_utilities, color = 'green', marker = 'o')
# #Add text to the markers for beta
# for i in range(len(utilities)):
# 	plt.text(social_welfare[i], min_utilities[i], str(betas[i]))
# plt.title("Tradeoff between social welfare and min utility")
# plt.xlabel("Social Welfare")
# plt.ylabel("Min Utility")
# plt.savefig("matthew_tradeoff2.png")
# plt.close()

# #Save the results
# import pandas as pd
# df = pd.DataFrame({'Beta':betas, 'Average Utility':utilities, 'Fairness':fairness, 'Social Welfare':social_welfare, 'CofVar':CofVar, 'Min Utility':min_utilities, 'Max Utility':max_utilities})
# df.to_csv("matthew_results.csv", index=False)


