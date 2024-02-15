import numpy as np
import tensorflow as tf

class EpsilonDecay():
	"""
	Wrapper class to handle epsilon decay
	"""
	def __init__(self, start, end, decay_rate, greedy=False):
		self.start = start
		self.end = end
		self.decay_rate = decay_rate
		self.current = start
		self.greedy = greedy

		if self.greedy:
			self.current = 0
	
	def get(self):
		return self.current
	
	def reset(self):
		self.current = self.start
		if self.greedy:
			self.current = 0
		return self.current
	
	def decay(self, eps=None):
		if eps is None:
			self.current = max(self.end, self.current*self.decay_rate)
		else:
			self.current = eps
		return self.current


def get_distance(a,b):
	#Get distance between two points
	return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)


def SI_reward(utils, direction='both'):
	#normalize
	avg = np.mean(utils)
	utils = [u/avg for u in utils]
	#Simple incentive/penalty for fairness
	if direction=='both':
		return [np.mean(utils) - utils[i] for i in range(len(utils))]
	elif direction=='adv':
		return [min(0, np.mean(utils) - utils[i]) for i in range(len(utils))]
	elif direction=='dis':
		return [max(0, np.mean(utils) - utils[i]) for i in range(len(utils))]
	else:
		print('Invalid direction')
		return None

def variance_penalty(utils):
	#normalize
	avg = np.mean(utils)
	if avg==0:
		return [0 for i in range(len(utils))]
	utils = [u/avg for u in utils]
	#Penalty for variance
	return [-np.var(utils) for i in range(len(utils))]

def get_fairness_from_su(su_prev, su_post, ftype="variance", action=None):
	#Compute the fairness score for each agent given previous utils and post utils
	# su_prev and su_post are lists of discounted utilities
	n_agents = len(su_prev)
	# ftype is the type of fairness to compute. Currently, only variance is supported
	if ftype=="variance":
		return [-np.var(su_post)/n_agents for i in range(n_agents)]
	elif ftype=="variance_diff":
		return [-np.var(su_post)/n_agents + np.var(su_prev)/n_agents for i in range(n_agents)]
	elif ftype=="split_diff":
		#Each agent gets score based on how much they contributed
		# su_post = [su_prev[i] + int(bool(action[i]+1)) for i in range(n_agents)] # This is only true for the mathhew envt. 
		scores = [0 for i in range(n_agents)]
		zbar = np.mean(su_prev)
		z2bar = np.mean(su_post)
		for i in range(n_agents):
			z_i = su_prev[i]
			z_i2 = su_post[i]
			scores[i] = -((z_i2 -z2bar)**2 - (z_i - zbar)**2)/n_agents   #The exact contribution. Not an estimate.
		return scores
	elif ftype=='SI':
		# su_post = [su_prev[i] + int(bool(action[i]+1)) for i in range(n_agents)] 
		zbar = np.mean(su_prev)
		scores = [0 for i in range(n_agents)]
		for i in range(n_agents):
			z_i = su_prev[i]
			z2_i = su_post[i]
			scores[i] = 2*(z_i - zbar)*(z2_i - z_i)
			# scores[i] = 2*(z_i - zbar)*(action[i])
		return scores
	else:
		print("Fairness type not supported. Exiting")
		exit()


#################### Logging ####################
def add_epi_metrics_to_logs(summary_writer, rewards, losses, beta, i_episode, max_steps, verbose=False, prefix="", logging=True):
	#Add episode's metrics to logs
	#rewards is a list of utilities for each agent
	variance = np.var(rewards)
	utility = np.sum(rewards)
	min_utility = min(rewards)
	fairness = -variance/(np.mean(rewards)+0.0001)
	objective = utility - beta*variance

	if verbose:
		print(rewards)
		print("Ep {:>5d} | Objective   {:>5.2f} | Beta {:>5.4f}".format(i_episode, objective, beta))
		print("Ep {:>5d} | Utility     {:>5.2f} | Variance {:>5.2f}".format(i_episode, utility, variance))
		print("Ep {:>5d} | Min Utility {:>5.2f} | Fairness {:>5.2f}".format(i_episode, min_utility, fairness))

	if logging:
		with summary_writer.as_default():
			tf.summary.scalar(prefix+"Utility", float(utility), step=i_episode)
			tf.summary.scalar(prefix+"Fairness", float(fairness), step=i_episode)
			tf.summary.scalar(prefix+"Min Utility", float(min_utility), step=i_episode)
			tf.summary.scalar(prefix+"Variance", float(variance), step=i_episode)
			tf.summary.scalar(prefix+"Objective", float(objective), step=i_episode)
			
			if losses is not None:
				for key, value in losses.items():
					tf.summary.scalar(prefix+key, float(value), step=i_episode)
	
	metrics = {'utility': utility, 'fairness': fairness, 'min_utility': min_utility, 'variance': variance, 'objective': objective}
	return metrics
		

def add_metric_to_logs(summary_writer, metric, name, i_episode, verbose=False, logging=True):
	if verbose:
		print(name, metric)
	if logging:
		with summary_writer.as_default():
			tf.summary.scalar(name, float(metric), step=i_episode)

def get_metrics_from_rewards(rewards, beta):
	#rewards is a list of utilities for each agent
	variance = np.var(rewards)
	utility = np.sum(rewards)
	min_utility = min(rewards)
	fairness = -variance/(np.mean(rewards)+0.0001)
	objective = utility - beta*variance
	metrics = {
		'utility': utility,
		'fairness': fairness,
		'min_utility': min_utility,
		'variance': variance,
		'objective': objective
	}
	return metrics

