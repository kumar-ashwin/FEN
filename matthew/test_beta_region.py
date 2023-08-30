"""
Load Split DDQN models, test them around their trained beta point and see generalization
"""
import os, sys, time  
import numpy as np
import tensorflow as tf
import random
import pandas as pd

from tensorflow.python.keras import backend as KTF
from keras import backend as K
import copy
import matplotlib.pyplot as plt

from tensorboard.plugins.hparams import api as hp

from Agents import DDQNAgent, SplitDDQNAgent, MultiHeadDDQNAgent
from matching import compute_best_actions
from utils import SI_reward, variance_penalty, get_fairness_from_su, EpsilonDecay, get_metrics_from_rewards, add_epi_metrics_to_logs, add_metric_to_logs
from Environment import MatthewEnvt

from tqdm import tqdm

# Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

greedy = True
training = False
reallocate = False
central_rewards = False
simple_obs = False
logging = False

split = True
learn_fairness = True
learn_utility = True
multi_head = True

phased_training = False

if not split:
	print("Unsupported")
	exit()

SI_beta = 0
learning_beta = 0.0
fairness_type = "split_diff" # ['split_diff', 'split_variance', 'variance_diff', 'variance', 'SI']
# fairness_type = "variance_diff" # ['split_diff', 'split_variance', 'variance_diff', 'variance', 'SI']

max_size = 0.5

mode = "Reallocate" if reallocate else "Fixed"
mode += "Central" if central_rewards else ""
mode += "" if simple_obs else "Complex"
mode+= "/Test/" #Only for testing new features
network_type = ""
if multi_head:
	network_type = "/MultiHead"
elif split and not multi_head: 
	network_type = "/Split"
elif not split:
	network_type = "/Joint"
mode+= network_type
mode += "Phased" if phased_training else ""
if split:
	mode += "NoUtility" if not learn_utility else ""
	mode += "NoFairness" if not learn_fairness else ""
mode += "/"
mode += f"{fairness_type}"

summary_writer = None

n_agents=10
n_resources=3

warm_start = 50
past_discount = 0.995
#initialize environments
M_test = MatthewEnvt(n_agents=n_agents, n_resources=n_resources, max_size=max_size, reallocate=reallocate, simple_obs=simple_obs, warm_start=warm_start, past_discount=past_discount)
M_train = MatthewEnvt(n_agents=n_agents, n_resources=n_resources, max_size=max_size, reallocate=reallocate, simple_obs=simple_obs, warm_start=warm_start, past_discount=past_discount)

config = tf.compat.v1.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.compat.v1.Session(config=config)
KTF.set_session(session)
T = 100   # learning frequency for the policy and value networks
totalTime = 0
GAMMA = 0.98
n_episode = 50
max_steps = 500
i_episode = 0
render = False

eps = EpsilonDecay(start=0.5, end=0.05, decay_rate=0.995, greedy=greedy)
ep_epsilon = eps.reset()

obs = M_test.get_obs()
num_features = len(obs[0][0])


learning_rate = 0.00003

agent = DDQNAgent(M_train, num_features, hidden_size=256, learning_rate=learning_rate, replay_buffer_size=250000, GAMMA=GAMMA, learning_beta=learning_beta)
if split:
	if multi_head:
		agent = MultiHeadDDQNAgent(M_train, num_features, hidden_size=256, learning_rate=learning_rate, replay_buffer_size=250000, GAMMA=GAMMA, learning_beta=learning_beta,
			    learn_utility=learn_utility, learn_fairness=learn_fairness, phased_learning=phased_training)
	else:
		agent = SplitDDQNAgent(M_train, num_features, hidden_size=256, learning_rate=learning_rate, replay_buffer_size=250000, GAMMA=GAMMA, learning_beta=learning_beta,
		learn_utility=learn_utility, learn_fairness=learn_fairness)

learning_beta=(None, None) #Just so this isn't accidentally used
fairness_model_folder = f"Models/DDQN/{mode}"
fairness_models = {}
utility_models = {}
# fairness_model_folder = 'Models/DoubleDQN/FixedComplex_split_diff'
#Open directory, get all folders, sort by beta, load models
for folder in os.listdir(fairness_model_folder):
	beta = float(folder)
	subfolders = os.listdir(fairness_model_folder+"/"+folder)
	m = subfolders[0]
	# model_loc = fairness_model_folder+"/"+folder+"/"+m+"/model_5000.ckpt"
	model_loc = fairness_model_folder+"/"+folder+"/"+m+"/best/best_model.ckpt"
	#load best model log. If best model is before 1500, load model_2000.ckpt
	log_loc = fairness_model_folder+"/"+folder+"/"+m+"/best/best_log.txt"
	with open(log_loc, 'r') as f:
		for line in f:
			if "Episode" in line:
				best_ep = int(line.split(": ")[1])
				break
	if best_ep<1000:
		model_loc = fairness_model_folder+"/"+folder+"/"+m+"/model_2000.ckpt"
	
	print(best_ep, beta)
	fairness_models[beta] = model_loc
	if split and not multi_head:
		fairness_models[beta] = model_loc+"_fair"
		if learn_utility:
			utility_models[beta] = model_loc+"_util"
		else:
			utility_models[beta] = "Models/DDQN/FixedComplex/Split/split_diff/0.0/1691557008/best/best_model.ckpt_util"

#sort by beta
fairness_models = {k: v for k, v in sorted(fairness_models.items(), key=lambda item: item)}
utility_models = {k: v for k, v in sorted(utility_models.items(), key=lambda item: item)}

sp = "Split" if split else ""
if split and multi_head:
	sp = "MultiHead"
both = "_both" if learn_utility and learn_fairness else ""
savename = f"Results/BetaSweep/{sp}DDQN_{fairness_type}{both}_bestm_{n_episode}.csv"
print(savename)

results = pd.DataFrame(columns=['utility', 'fairness', 'min_utility', 'objective', 'variance', 'beta'])
for fairness_beta, model_loc in fairness_models.items():
	if multi_head:
		agent.load_model(model_loc)
	else:
		agent.load_util_model(utility_models[fairness_beta])
		agent.load_fair_model(model_loc)
	base_learning_beta = fairness_beta
	#Create a search space of learning betas from /10 to *10
	evaluation_betas = [base_learning_beta/10, base_learning_beta/5, base_learning_beta/2, base_learning_beta, base_learning_beta*2, base_learning_beta*5, base_learning_beta*10]
	evaluation_betas = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, base_learning_beta]
	evaluation_betas_lists = [[0],[0.001*(i+1) for i in range(9)], [0.01*(i+1) for i in range(9)], [0.1*(i+1) for i in range(9)], [1, 2, 5, 10, base_learning_beta]]
	evaluation_betas = []
	for arr in evaluation_betas_lists:
		evaluation_betas.extend(arr)
	evaluation_betas = list(set(evaluation_betas))
	evaluation_betas.sort()
	
	for eval_beta in evaluation_betas:
		# agent.learning_beta = eval_beta
		agent.set_beta(eval_beta)
		run_metrics = {'utility':[], 'fairness':[], 'min_utility':[], 'objective':[],'variance':[]}

		i_episode = 0
		for i_episode in tqdm(range(1, n_episode+1)):
			# i_episode+=1
			steps=0
			M_test.reset()
			obs = M_test.get_obs()
			ep_epsilon = eps.decay()
			while steps<max_steps:
				steps+=1
				actions = compute_best_actions(agent, obs, M_test.targets, n_agents, n_resources, M_test.discounted_su, beta=SI_beta, epsilon=0)
				rewards = M_test.step(actions)
				obs = M_test.get_obs()
			epi_metrics = add_epi_metrics_to_logs(summary_writer, M_test.su, None, eval_beta, i_episode, max_steps, verbose=False, prefix="", logging=logging)
			for key, value in epi_metrics.items():
				run_metrics[key].append(value)
		#print averages
		run_metrics['beta'] = [eval_beta for i in range(len(run_metrics['utility']))]
		run_metrics['base_beta'] = [base_learning_beta for i in range(len(run_metrics['utility']))]
		avg_results = pd.DataFrame({key:np.mean(value) for key, value in run_metrics.items()}, index=[0])
		print(avg_results)
		#Add to results
		results = pd.concat([results, avg_results], ignore_index=True)
		
		results.to_csv(savename)

print(results)
results.to_csv(savename)
#plot results
# import plotly.express as px
# #util vs fairness, colors = beta
# fig = px.scatter(results, x="utility", y="fairness", color="beta")
# fig.show()