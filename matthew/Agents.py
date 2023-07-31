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
