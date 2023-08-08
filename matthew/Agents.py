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


class SimpleValueNetwork():
	"""
		Just returns the discounted reward for each action
	"""
	def __init__(self, discount_factor=0.99, score_func=None):
		self.discount_factor = discount_factor
		self.score_func = score_func
	
	def get(self, states, score_func=None):
		if score_func is None:
			score_func = self.score_func
		#States is a list of lists
		utils = []
		for h in states:
			disc_reward = self.discount_factor**score_func(h)
			utils.append(disc_reward)
		if len(utils)==1:
			return utils[0]
		return utils

	def update(self, states, discounted_rewards):
		return 0

	def save_model(self, save_path):
		pass

	def load_model(self, save_path):
		pass

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
			# self.loss = tf.compat.v1.losses.mean_squared_error(self.output, self.rollout)
			#Huber loss
			self.loss = tf.compat.v1.losses.huber_loss(self.output, self.rollout)
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

	def set_weights(self, weights):
		with self.tf_graph.as_default():
			for i in range(len(self.W)):
				self.W[i].load(weights[i], self.session)
	
	def get_weights(self):
		with self.tf_graph.as_default():
			weights = self.session.run(self.W)
		return weights

class FairValueNetwork(ValueNetwork):
	def __init__(self, value_net, num_features, hidden_size, learning_rate=.01, beta=0.5):
		#initialize the super class
		super(FairValueNetwork, self).__init__(num_features, hidden_size, learning_rate)
		self.value_net = value_net
		self.beta = beta
	
	def get(self, states):
		utility = self.value_net.get(states)
		fairness = self.session.run(self.output, feed_dict={self.observations: states})
		
		#combine the two
		return utility + self.beta*fairness
	
	def get_fairness_value(self, states):
		return self.session.run(self.output, feed_dict={self.observations: states})
	
	def get_utility_value(self, states):
		return self.value_net.get(states)


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
			


#code a replay buffer
class ReplayBuffer:
	def __init__(self, max_size=100000):
		self.buffer = []
		self.max_size = max_size
		self.size = 0

	def add(self, experience):
		self.buffer.append(experience)
		self.size += 1
		if self.size > self.max_size:
			self.buffer.pop(0)
			self.size -= 1
	
	def add_batch(self, experiences):
		self.buffer.extend(experiences)
		self.size += len(experiences)
		while self.size > self.max_size:
			self.buffer.pop(0)
			self.size -= 1
		

	def sample(self, batch_size):
		samples = random.sample(self.buffer, batch_size)
		# return map(list, zip(*samples))
		return samples

	def size(self):
		return self.size

	def clear(self):
		self.buffer = []
		self.size = 0

#Prioritized replay buffer
class PrioritizedReplayBuffer(ReplayBuffer):
	def __init__(self, max_size=100000):
		super(PrioritizedReplayBuffer, self).__init__(max_size=max_size)
		self.priorities = np.zeros((max_size,), dtype=np.float32)
		
	def add(self, experience):
		max_prio = self.priorities.max() if self.buffer else 1.0
		super().add(experience)
		self.priorities[self.size-1] = max_prio

	def sample(self, batch_size, beta=0.4):
		if self.size == self.max_size:
			prios = self.priorities
		else:
			prios = self.priorities[:self.size]
		probs = prios ** beta
		probs /= probs.sum()
		indices = np.random.choice(self.size, batch_size, p=probs)
		samples = [self.buffer[idx] for idx in indices]
		total = self.size
		weights = (total * probs[indices]) ** (-beta)
		weights /= weights.max()
		return samples, indices, weights
	
	def update_priorities(self, batch_indices, batch_priorities):
		for idx, prio in zip(batch_indices, batch_priorities):
			self.priorities[idx] = prio

	def clear(self):
		super().clear()
		self.priorities = np.zeros((self.max_size,), dtype=np.float32)


from matching import compute_best_actions

class DDQNAgent():
	def __init__(self, 
	      envt,
	      num_features, 
		  hidden_size=256, 
		  learning_rate=0.0001, 
		  replay_buffer_size=100000,
		  GAMMA=0.99,
		  learning_beta=0.0,
		  ):
		
		self.envt = envt
		self.GAMMA = GAMMA
		self.learning_beta = learning_beta

		self.VF1 = ValueNetwork(num_features, hidden_size, learning_rate)
		self.TargetNetwork1 = ValueNetwork(num_features, hidden_size, learning_rate)
		self.TargetNetwork1.set_weights(self.VF1.get_weights())
		self.replayBuffer1 = ReplayBuffer(replay_buffer_size)

		self.VF2 = ValueNetwork(num_features, hidden_size, learning_rate)
		self.TargetNetwork2 = ValueNetwork(num_features, hidden_size, learning_rate)
		self.TargetNetwork2.set_weights(self.VF2.get_weights())
		self.replayBuffer2 = ReplayBuffer(replay_buffer_size)

		self.nets = [self.VF1, self.VF2]
		self.RBs = [self.replayBuffer1, self.replayBuffer2]
		self.current_net_index = 0
		self.current_net = self.VF1
		self.current_RB = self.replayBuffer1

	def get(self, states):
		return self.current_net.get(states)
	
	def set_active_net(self, net_index):
		self.current_net_index = net_index
		self.current_net = self.nets[self.current_net_index]
		self.current_RB = self.RBs[self.current_net_index]

	def add_experience(self, post_decision_state, rewards, f_rewards, new_state):
		# experience = copy.deepcopy([post_decision_state, rewards, f_rewards, new_state])
		experience = {
			'pd_state': copy.deepcopy(post_decision_state),
			'rewards': copy.deepcopy(rewards),
			'f_rewards': copy.deepcopy(f_rewards),
			'new_state': copy.deepcopy(new_state)
		}
		self.current_RB.add(experience)
	
	def _update(self, experiences, net, target_net, double_target):
		#Abstracted function to handle updates for both networks
		# Note: SI beta not supported, always steps with beta=0
		n_agents = self.envt.n_agents
		n_resources = self.envt.n_resources
		losses = []
		for experience in experiences:
			#Compute best action
			pd_state, rewards, f_rewards, new_state = experience['pd_state'], experience['rewards'], experience['f_rewards'], experience['new_state']
			self.envt.set_state(new_state)
			succ_obs = self.envt.get_obs()
			opt_actions = compute_best_actions(target_net, succ_obs, self.envt.targets, n_agents, n_resources, self.envt.discounted_su, beta=0, epsilon=0.0)
			new_pd_states = self.envt.get_post_decision_states(succ_obs, opt_actions)
			
			td_rewards = np.array(rewards)
			if self.learning_beta>0:
				td_rewards = rewards + self.learning_beta*np.array(f_rewards)

			#Perform batched updates
			states = np.array([pd_state[i] for i in range(n_agents)])
			target_values = np.array([td_rewards[i] + self.GAMMA * double_target.get(np.array([new_pd_states[i]])) for i in range(n_agents)])
			target_values = target_values.reshape(-1)
			
			loss = net.update(states, target_values)
			losses.append(loss)
		return losses
	
	def update(self, num_samples, num_min_samples=100000):
		min_RB_size = num_min_samples//self.envt.n_agents
		if self.replayBuffer1.size < min_RB_size or self.replayBuffer2.size < min_RB_size:
			return	[],[]
		
		self.envt.reset()
		experiences1 = self.replayBuffer1.sample(num_samples)
		experiences2 = self.replayBuffer2.sample(num_samples)

		losses1 = self._update(experiences1, self.VF1, self.TargetNetwork1, self.TargetNetwork2)
		losses2 = self._update(experiences2, self.VF2, self.TargetNetwork2, self.TargetNetwork1)
		return losses1, losses2

	def update_from_experiences(self, experiences1, experiences2):
		# For use from an external source
		self.envt.reset()
		losses1 = self._update(experiences1, self.VF1, self.TargetNetwork1, self.TargetNetwork2)
		losses2 = self._update(experiences2, self.VF2, self.TargetNetwork2, self.TargetNetwork1)
		return losses1, losses2

	
	def update_target_networks(self):
		self.TargetNetwork1.set_weights(self.VF1.get_weights())
		self.TargetNetwork2.set_weights(self.VF2.get_weights())

	def load_model(self, save_path):
		self.VF1.load_model(save_path)
		self.VF2.load_model(save_path)
		self.TargetNetwork1.load_model(save_path)
		self.TargetNetwork2.load_model(save_path)

	def save_model(self, save_path):
		self.VF1.save_model(save_path)

	def get_weights(self):
		return self.VF1.get_weights()
	
	def set_weights(self, weights):
		self.VF1.set_weights(weights)
		self.VF2.set_weights(weights)
		self.TargetNetwork1.set_weights(weights)
		self.TargetNetwork2.set_weights(weights)


class SplitDDQNAgent():
	"""
	Agent class for Split DDQN
	A wrapper class for the two DDQNs, one for fairness, one for efficiency
	Handles saving experiences and dividing up the learning between the two networks
	Both agents will see the same state, but will have different rewards. 
		Fair agent will see the fairness rewards, and the util agent will see the utility rewards
		The experiences are preprocessed before sending to the networks
	"""
	def __init__(self, 
	      envt,
	      num_features, 
		  hidden_size=256, 
		  learning_rate=0.0001, 
		  replay_buffer_size=100000,
		  GAMMA=0.99,
		  learning_beta=0.0,
		  learn_utility=True,
		  learn_fairness=True,
		  ):
		self.envt = envt
		self.learning_beta = learning_beta
		self.learn_utility = learn_utility
		self.learn_fairness = learn_fairness
		self.GAMMA = GAMMA

		self.utilAgent = DDQNAgent(envt, num_features, hidden_size, learning_rate, replay_buffer_size=0, GAMMA=GAMMA, learning_beta=0)
		self.fairAgent = DDQNAgent(envt, num_features, hidden_size, learning_rate, replay_buffer_size=0, GAMMA=GAMMA, learning_beta=0)
		self.RB1 = ReplayBuffer(replay_buffer_size)
		self.RB2 = ReplayBuffer(replay_buffer_size)

		self.current_net_index = 0
		#select this index for both util and fair agent
		self.utilAgent.set_active_net(self.current_net_index)
		self.fairAgent.set_active_net(self.current_net_index)
		self.current_RB = self.RB1
		
	def set_active_net(self, net_index):
		self.current_net_index = net_index
		self.utilAgent.set_active_net(net_index)
		self.fairAgent.set_active_net(net_index)
		self.current_RB = self.RB1 if net_index==0 else self.RB2
	
	def get(self, state, beta=None):
		if beta is None:
			beta = self.learning_beta
		return self.utilAgent.get(state) + beta*self.fairAgent.get(state)

	def get_utility(self, state):
		return self.utilAgent.get(state)
	
	def get_fairness(self, state):
		return self.fairAgent.get(state)
	
	def add_experience(self, post_decision_state, rewards, f_rewards, new_state):
		# experience = copy.deepcopy([post_decision_state, rewards, f_rewards, new_state])
		experience = {
			'pd_state': copy.deepcopy(post_decision_state),
			'rewards': copy.deepcopy(rewards),
			'f_rewards': copy.deepcopy(f_rewards),
			'new_state': copy.deepcopy(new_state)
		}
		self.current_RB.add(experience)

	def update(self, num_samples, num_min_samples=100000):
		min_RB_size = num_min_samples//self.envt.n_agents
		loss_logs = {'util': [], 'fair': []}
		if self.RB1.size < min_RB_size or self.RB2.size < min_RB_size:
			return	loss_logs
		
		self.envt.reset()
		experiences1 = self.RB1.sample(num_samples)
		experiences2 = self.RB2.sample(num_samples)

		if self.learn_utility:
			u_losses1, u_losses2= self.utilAgent.update_from_experiences(experiences1, experiences2)
			loss_logs['util'].append(u_losses1)
			loss_logs['util'].append(u_losses2)

		if self.learn_fairness:
			f_experience1 = []
			#Modify experiences to have fairness rewards
			for experience in experiences1:
				f_experience = copy.deepcopy(experience)
				f_experience['rewards'] = f_experience['f_rewards']
				f_experience1.append(f_experience)
			f_experience2 = []
			for experience in experiences2:
				f_experience = copy.deepcopy(experience)
				f_experience['rewards'] = f_experience['f_rewards']
				f_experience2.append(f_experience)
			
			f_losses1, f_losses2 = self.fairAgent.update_from_experiences(f_experience1, f_experience2)
			loss_logs['fair'].append(f_losses1)
			loss_logs['fair'].append(f_losses2)
		
		return loss_logs
	
	def update_target_networks(self):
		self.utilAgent.update_target_networks()
		self.fairAgent.update_target_networks()

	def load_util_model(self, save_path):
		self.utilAgent.load_model(save_path)

	def load_fair_model(self, save_path):
		self.fairAgent.load_model(save_path)

	def save_util_model(self, save_path):
		self.utilAgent.save_model(save_path)

	def save_fair_model(self, save_path):
		self.fairAgent.save_model(save_path)

	def save_model(self, save_path):
		if self.learn_utility:
			self.utilAgent.save_model(save_path+'_util')
		if self.learn_fairness:
			self.fairAgent.save_model(save_path+'_fair')
		