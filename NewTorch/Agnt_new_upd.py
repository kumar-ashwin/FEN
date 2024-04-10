"""
Torch versions of the agents.
Adding PPO and A2C classes, needs experimentation
"""
import sys
import copy
import torch
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer

class metaNetwork(nn.Module):
	"""
	Boilerplate functions
	"""
	def __init__(self):
		super(metaNetwork, self).__init__()
	
	def set_weights(self, weights):
		self.load_state_dict(weights)
	
	def get_weights(self):
		return self.state_dict()
	
	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		self.load_state_dict(torch.load(path))

class ValueNetwork(metaNetwork):
	"""
	A value network that predicts the value of a state.
	"""
	def __init__(self, num_features, hidden_size):
		super(ValueNetwork, self).__init__()
		self.fc1 = nn.Linear(num_features, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class PPOValueNetwork(metaNetwork):
	"""
	A PPO network that outputs a value.
	Predict a mean and variance, and sample from a normal distribution.
	"""
	def __init__(self, num_features, hidden_size):
		super(PPOValueNetwork, self).__init__()
		self.fc1 = nn.Linear(num_features, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.mean = nn.Linear(hidden_size, 1)
		self.var = nn.Linear(hidden_size, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		mean = self.mean(x)
		var = F.softplus(self.var(x)) # Ensure variance is positive
		return mean, var
	

class A2CNetwork(metaNetwork):
	"""
	A2C implementation. 
	"""
	def __init__(self, num_features, hidden_size, num_actions):
		super(A2CNetwork, self).__init__()
		self.fc1 = nn.Linear(num_features, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.mean = nn.Linear(hidden_size, num_actions)
		self.var = nn.Linear(hidden_size, num_actions)
		self.value = nn.Linear(hidden_size, 1)
	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		means = self.mean(x)
		vars = F.softplus(self.var(x))
		value = self.value(x)
		return means, vars, value


class PolicyNetwork(metaNetwork):
	"""
	A policy network that outputs a probability distribution over actions.
	"""
	def __init__(self, num_features, hidden_size, num_actions):
		super(PolicyNetwork, self).__init__()
		self.fc1 = nn.Linear(num_features, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_actions)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.softmax(x, dim=1)

class MultiHeadValueNetwork(metaNetwork):
	"""
	Value network with two heads, one for fairness and one for utility
	"""
	def __init__(self, num_features, hidden_size, learning_beta=0.0):
		super(MultiHeadValueNetwork, self).__init__()
		self.learning_beta = learning_beta
		self.eval_beta = learning_beta
		self.fc1 = nn.Linear(num_features, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fairness_head = nn.Linear(hidden_size, 1)
		self.utility_head = nn.Linear(hidden_size, 1)
	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		fairness = self.fairness_head(x)
		utility = self.utility_head(x)
		mult = 0 if self.learning_beta == 0 else self.eval_beta/self.learning_beta
		return utility + mult*fairness
	
	def get_util(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		utility = self.utility_head(x)
		return utility
	
	def get_fair(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		fairness = self.fairness_head(x)
		return fairness
	

"""
Agents:
"""

class Agent:
	def __init__(
			self,
			env,
			num_features,
			hidden_size,
			num_actions,
			learning_rate=0.001,
			GAMMA=0.99,
			learning_beta=0.0,
	):
		self.env = env
		self.num_features = num_features
		self.hidden_size = hidden_size
		self.num_actions = num_actions
		self.learning_rate = learning_rate
		self.GAMMA = GAMMA
		self.learning_beta = learning_beta
		self.eval_beta = learning_beta

	def get(self, state):
		pass

	def update(self):
		pass

class DoubleDQNAgent(Agent):
	"""
	This agent uses a value network to predict Q values of post-decision states.
	Uses double DQN to stabilize learning.
	Key idea: during the update, pick a* = argmax_a Q(s', a) using the online network,
	but calculate the target value using the target network Q'(s',a*).
	YDoubleDQN_t = R_(t+1) + Q(S_(t+1), argmax_a Q(St+1 a; w) w_t ) 
	Follows: Deep Reinforcement Learning with Double Q-learning (Hasselt et al. 2015)
	"""
	def __init__(
		self,
		env,
		num_features,
		hidden_size,
		learning_rate=0.001,
		replay_buffer_size=1000000,
		GAMMA=0.99,
		learning_beta=0.0,
	):
		super(DoubleDQNAgent, self).__init__(env, num_features, hidden_size, None, learning_rate, GAMMA, learning_beta)

		self.replay_buffer = ReplayBuffer(replay_buffer_size)
		self.q_network = ValueNetwork(num_features, hidden_size)
		self.target_q_network = ValueNetwork(num_features, hidden_size)
		self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
		self.loss = nn.MSELoss()

	def get(self, state, target=False):
		state = Variable(torch.from_numpy(state).float().unsqueeze(0))
		if target:
			return self.target_q_network(state)
		return self.q_network(state)
	
	def gets(self, states, target=False):
		states = Variable(torch.from_numpy(states).float())
		if target:
			return self.target_q_network(states)
		return self.q_network(states)

	def add_experience(self, post_decision_state, rewards, f_rewards, new_state, done):
		experience = {
			'pd_state': copy.deepcopy(post_decision_state),
			'rewards': copy.deepcopy(rewards),
			'f_rewards': copy.deepcopy(f_rewards),
			'new_state': copy.deepcopy(new_state),
			'done': done,
		}
		self.replay_buffer.add(experience)
	
	def _update(self, experiences):
		losses = []
		for experience in experiences:
			pd_state, rewards, f_rewards, new_state, done = experience['pd_state'], experience['rewards'], experience['f_rewards'], experience['new_state'], experience['done']
			n_agents = len(pd_state)
			self.env.set_state(new_state)
			succ_obs = self.env.get_obs()

			# Compute the optimal actions using the online q-network
			opt_actions = self.env.compute_best_actions(self, self.env, succ_obs)
			new_pd_states = self.env.get_post_decision_states(succ_obs, opt_actions)

			td_rewards = np.array(rewards)
			td_f_rewards = np.array(f_rewards)
			total_rewards = td_rewards + self.learning_beta * td_f_rewards

			states = np.array([pd_state[i] for i in range(n_agents)])
			
			values = self.gets(states).squeeze()
			target_values = total_rewards + (int(not(done)))*self.GAMMA * self.gets(np.array(new_pd_states), target=True).detach().numpy().flatten()
			target_values = Variable(torch.from_numpy(target_values).float())
			
			loss = self.loss(values, target_values)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			losses.append(loss.item())
		return losses

	def update(self, num_samples, num_min_samples=100000):
		if self.replay_buffer.size < num_min_samples:
			return []
		self.env.reset()
		experiences = self.replay_buffer.sample(num_samples)
		
		losses = self._update(experiences)
		return losses
	
	def update_from_experience(self, experiences):
		self.env.reset()
		return self._update(experiences)

	def update_target_network(self):
		self.target_q_network.set_weights(self.q_network.get_weights())

	def save_model(self, path):
		self.q_network.save(path)

	def load_model(self, path):
		self.q_network.load(path)
		self.update_target_network()

class SplitDoubleDQNAgent(Agent):
	def __init__(
		self,
		env,
		num_features,
		hidden_size,
		learning_rate=0.001,
		replay_buffer_size=1000000,
		GAMMA=0.99,
		learning_beta=0.0,
		learn_fairness=True,
		learn_utility=True,
	):
		super(SplitDoubleDQNAgent, self).__init__(env, num_features, hidden_size, None, learning_rate, GAMMA, learning_beta)
		self.replay_buffer = ReplayBuffer(replay_buffer_size)
		self.utilAgent = DoubleDQNAgent(env, num_features, hidden_size, learning_rate, replay_buffer_size, GAMMA, learning_beta=0)
		self.fairAgent = DoubleDQNAgent(env, num_features, hidden_size, learning_rate, replay_buffer_size, GAMMA, learning_beta=0)
		self.learn_fairness = learn_fairness
		self.learn_utility = learn_utility
		

	def get(self, state, beta=None):
		if beta is None:
			beta = self.learning_beta
		return self.utilAgent.get(state) + beta*self.fairAgent.get(state)
	
	def add_experience(self, post_decision_state, rewards, f_rewards, new_state, done):
		self.replay_buffer.add({
			'pd_state': post_decision_state,
			'rewards': rewards,
			'f_rewards': f_rewards,
			'new_state': new_state,
			'done': done,
		})
	
	def update(self, num_samples, num_min_samples=100000):
		loss_logs = {'util': [], 'fair': []}
		if self.replay_buffer.size < num_min_samples:
			return loss_logs
		
		self.env.reset()
		experiences = self.replay_buffer.sample(num_samples)
		loss_logs['util'] = self.utilAgent.update_from_experience(experiences)
		fair_experiences = []
		for experience in experiences:
			f_exp = copy.deepcopy(experience)
			f_exp['rewards'] = f_exp['f_rewards']
			fair_experiences.append(f_exp)
		loss_logs['fair'] = self.fairAgent.update_from_experience(fair_experiences)

		return loss_logs

	def update_target_network(self):
		self.utilAgent.update_target_network()
		self.fairAgent.update_target_network()

	def save_model(self, path):
		self.utilAgent.save_model(path + "_util")
		self.fairAgent.save_model(path + "_fair")

	def load_util_model(self, path):
		self.utilAgent.load_model(path)
		self.utilAgent.update_target_network()
	
	def load_fair_model(self, path):
		self.fairAgent.load_model(path)
		self.fairAgent.update_target_network()


class MultiHeadDoubleDQNAgent(Agent):
	"""
	Agent with 2 heads, one for fairness and one for utility.
	Uses Double DQN to stabilize learning.
	Does not support phased learning.
	"""
	def __init__(
			self, env, num_features, hidden_size, learning_rate=0.001, replay_buffer_size=1000000, GAMMA=0.99, learning_beta=0.0,
			learn_fairness=True, learn_utility=True, phased_learning=False			
	):
		super(MultiHeadDoubleDQNAgent, self).__init__(env, num_features, hidden_size, None, learning_rate, GAMMA, learning_beta)
		self.replay_buffer = ReplayBuffer(replay_buffer_size)
		self.q_network = MultiHeadValueNetwork(num_features, hidden_size, learning_beta)
		self.target_q_network = MultiHeadValueNetwork(num_features, hidden_size, learning_beta)
		self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
		self.loss_util = nn.MSELoss()
		self.loss_fair = nn.MSELoss()
		self.learn_fairness = learn_fairness
		self.learn_utility = learn_utility
		if phased_learning:
			raise Exception("Phased learning not supported for MultiHeadDoubleDQNAgent")
	
	def set_eval_beta(self, beta):
		self.eval_beta = beta
		self.q_network.eval_beta = beta
		self.target_q_network.eval_beta = beta
	
	def set_learning_beta(self, beta):
		self.learning_beta = beta
		self.q_network.learning_beta = beta
		self.target_q_network.learning_beta = beta

	def get(self, state, target=False):
		state = Variable(torch.from_numpy(state).float().unsqueeze(0))
		if target:
			return self.target_q_network(state)
		return self.q_network(state)
	
	def gets(self, states, target=False):
		states = Variable(torch.from_numpy(states).float())
		if target:
			return self.target_q_network(states)
		return self.q_network(states)
	
	def get_fair(self, state, target=False):
		state = Variable(torch.from_numpy(state).float().unsqueeze(0))
		if target:
			return self.target_q_network.get_fair(state)
		return self.q_network.get_fair(state)
	
	def get_util(self, state, target=False):
		state = Variable(torch.from_numpy(state).float().unsqueeze(0))
		if target:
			return self.target_q_network.get_util(state)
		return self.q_network.get_util(state)
	
	def add_experience(self, post_decision_state, rewards, f_rewards, new_state, done):
		experience = {
			'pd_state': copy.deepcopy(post_decision_state),
			'rewards': copy.deepcopy(rewards),
			'f_rewards': copy.deepcopy(f_rewards),
			'new_state': copy.deepcopy(new_state),
			'done': done,
		}
		self.replay_buffer.add(experience)

	# def update(self, fairness_error, utility_error):
	#     # Scale the fairness error
	#     scaled_fairness_error = fairness_error * self.learning_beta

	#     # Combine the errors
	#     total_error = scaled_fairness_error + utility_error

	#     # Zero the gradients
	#     self.optimizer.zero_grad()

	#     # Perform backpropagation
	#     total_error.backward()

	#     # Update the weights
	#     self.optimizer.step()
	# def update(self, fairness_target, utility_target):
	# 	# Zero the gradients
	# 	self.optimizer.zero_grad()

	# 	# Calculate the fairness loss and backpropagate
	# 	fairness_output = self.get_fair()
	# 	fairness_loss = self.loss(fairness_output, fairness_target * self.learning_beta)
	# 	fairness_loss.backward(retain_graph=True)

	# 	# Calculate the utility loss and backpropagate
	# 	utility_output = self.get_util()
	# 	utility_loss = self.loss(utility_output, utility_target)
	# 	utility_loss.backward()

	# 	# Update the weights
	# 	self.optimizer.step()
	def _update_basic(self, experiences):
		u_losses = []
		f_losses = []
		for experience in experiences:
			pd_state, rewards, f_rewards, new_state, done = experience['pd_state'], experience['rewards'], experience['f_rewards'], experience['new_state'], experience['done']
			n_agents = len(pd_state)
			self.env.set_state(new_state)
			succ_obs = self.env.get_obs()

			# Compute the optimal actions using the online q-network
			opt_actions = self.env.compute_best_actions(self, self.env, succ_obs)
			new_pd_states = self.env.get_post_decision_states(succ_obs, opt_actions)

			td_rewards = np.array(rewards)
			td_f_rewards = np.array(f_rewards)
			total_rewards = td_rewards + self.learning_beta * td_f_rewards

			states = np.array([pd_state[i] for i in range(n_agents)])

			values = self.gets(states).squeeze()
			target_values = total_rewards + (int(not(done)))*self.GAMMA * self.gets(np.array(new_pd_states), target=True).detach().numpy().flatten()
			target_values = Variable(torch.from_numpy(target_values).float())
			loss = self.loss(values, target_values)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			u_losses.append(loss.item())
		return u_losses, f_losses
	
	def _update_split(self, experiences):
		# Backprop each head separately and return the losses
		#Possible solutionL Return two values from the forward method of MultiHeadValueNetwork
		u_losses = []
		f_losses = []

		for experience in experiences:
			pd_state, rewards, f_rewards, new_state, done = experience['pd_state'], experience['rewards'], experience['f_rewards'], experience['new_state'], experience['done']
			n_agents = len(pd_state)
			self.env.set_state(new_state)
			succ_obs = self.env.get_obs()

			# Compute the optimal actions using the online q-network
			opt_actions = self.env.compute_best_actions(self, self.env, succ_obs)
			new_pd_states = self.env.get_post_decision_states(succ_obs, opt_actions)

			td_rewards = np.array(rewards)
			td_f_rewards = np.array(f_rewards)

			states = np.array([pd_state[i] for i in range(n_agents)])

			values_util = self.get_util(states).squeeze()
			target_values_util = td_rewards + (int(not(done)))*self.GAMMA * self.get_util(np.array(new_pd_states), target=True).detach().numpy().flatten()
			target_values_util = Variable(torch.from_numpy(target_values_util).float())

			values_fair = self.get_fair(states).squeeze()
			target_values_fair = td_f_rewards + (int(not(done)))*self.GAMMA * self.get_fair(np.array(new_pd_states), target=True).detach().numpy().flatten()
			target_values_fair = Variable(torch.from_numpy(target_values_fair).float())
			
			loss_util = self.loss_util(values_util, target_values_util)
			loss_fair = self.loss_fair(values_fair, target_values_fair)
			total_loss = loss_util + self.learning_beta * loss_fair

			self.optimizer.zero_grad()
			total_loss.backward()
			self.optimizer.step()

			# self.optimizer.zero_grad()
			# loss_util.backward()
			# self.optimizer.step()
			

			# self.optimizer.zero_grad()
			# loss_fair.backward()
			# self.optimizer.step()

			u_losses.append(loss_util.item())
			f_losses.append(loss_fair.item())
		return u_losses, f_losses
	
	def update(self, num_samples, num_min_samples=100000):
		loss_logs = {'util': [], 'fair': []}
		if self.replay_buffer.size < num_min_samples:
			return loss_logs
		self.env.reset()
		experiences = self.replay_buffer.sample(num_samples)
		if self.learn_fairness and self.learn_utility:
			loss_logs['util'], loss_logs['fair'] = self._update_split(experiences)
			# loss_logs = 0
		else:
			loss_logs['util'], _ = self._update_basic(experiences)
		return loss_logs
		
	def update_from_experience(self, experiences):
		if self.learn_fairness and self.learn_utility:
			return self._update_split(experiences)
		return self._update_basic(experiences)
	
	def update_target_network(self):
		self.target_q_network.set_weights(self.q_network.get_weights())

	def save_model(self, path):
		self.q_network.save(path)

	def load_model(self, path):
		self.q_network.load(path)
		self.update_target_network()




class PPOValueAgent(Agent):
	"""
	Proximal Policy Optimization agent.
	Predicts a mean and variance for the value, and samples from a normal distribution.
	"""
	def __init__(self, env, num_features, hidden_size, num_actions, learning_rate=0.001, GAMMA=0.99, learning_beta=0):
		super().__init__(env, num_features, hidden_size, num_actions, learning_rate, GAMMA, learning_beta)
		


