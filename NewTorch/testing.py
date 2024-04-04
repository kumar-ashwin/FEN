"""
A general purpose testing script.
So far, only for Joint agents.
ToDo: Use the general validation function. 
- Do we need the step-by-step information?
"""
import os, time  
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import csv
from tqdm import tqdm

from agent_utils import get_agent, take_env_step, get_env, run_validation
from utils import get_metrics_from_rewards
from process_args import process_args

import matplotlib.pyplot as plt
import copy
# Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

args, train_args = process_args("WarmStart", load_default=True)
# Location to test

folder = "Models/WarmStartTestSimpleMultiHead/Simple/split_diff/MultiHead/"
folder = "Models/WarmStartTestFixedMultiHead/Simple/split_diff/MultiHead/"

#load a log file to get model arguments
beta = os.listdir(folder)[0]
timestamp = os.listdir(folder+beta)[0]
logfile = folder+beta+"/"+timestamp+"/best/best_log.txt"
targets = [args, train_args]

with open(logfile, "r") as f:
    for line in f:
        key, value = line.split(":")
        value = value.strip()
        try:
            type_ = type(getattr(args, key))
            target = args
        except AttributeError:
            try:
                type_ = type(getattr(train_args, key))
                target = train_args
            except AttributeError:
                continue
        if type_ == bool:
            value = value == "True"
        else:
            value = type_(value)
        setattr(target, key, value)

args.save_path = "results/"+args.save_path
print(args.save_path)

M, M_train, M_val = get_env(args.env_name, args, train_args)
M_val.external_trigger = True
obs = M.get_obs()
num_features = len(obs[0][0])
n_agents = len(obs[0])

def load_agent(model_path, beta):
    agent = get_agent(train_args, args.training, num_features, M_train)
    agent.load_model(model_path)
    agent.set_eval_beta(beta)
    agent.learning_beta = beta
    return agent


def validate_and_plot(agent, M_val, max_steps, args, n_agents):
    num_eps = 3
    all_resource_rates =  []

    #Run validation episodes with the current policy
    val_metrics = {'system_utility':[], 'fairness':[], 'min_utility':[], 'objective':[],'variance':[]}
    for val_eps in tqdm(range(num_eps)):
        resource_rates = [[] for _ in range(n_agents)]
        M_val.reset()
        obs = M_val.get_obs()
        score = 0
        for steps in range(max_steps):
            util, obs = take_env_step(M_val, agent, obs, steps, 0, args, add_to_replay=False)
            score+=util

            for i in range(n_agents):
                resource_rates[i].append(M_val.get_fairness_rewards()[i])

            if args.render:
                M_val.render()
                time.sleep(0.1)
                print(steps, val_eps)

        # print(M_val.su)
        metrics = get_metrics_from_rewards(M_val.su, args.learning_beta, fair_rewards=M_val.get_fairness_rewards())
        for key, value in metrics.items():
            val_metrics[key].append(value)
        
        all_resource_rates.append(resource_rates)

    # Average the resource rates
    resource_rates = np.mean(all_resource_rates, axis=0)
    # plot the resource rates
    import matplotlib.pyplot as plt
    xs = range(len(resource_rates[0]))
    for i in range(n_agents):
        plt.plot(xs, resource_rates[i], label="Agent {}".format(i))
    plt.legend()

    mean_val_metrics = {}
    for key, value in val_metrics.items():
        mean_val_metrics[key] = np.mean(value)
    
    return mean_val_metrics




betas = os.listdir(folder)
beta_vals = {float(beta): beta for beta in betas}
betas = sorted(betas, key=lambda x: float(x))

if False:
    all_metrics = {}
    for beta in betas:
        beta_val = float(beta)
        print("Evaluating beta ", beta_val)
        timestamp = os.listdir(folder+beta)[0]
        agent = load_agent(folder+beta+"/"+timestamp+"/best/best_model.ckpt", float(beta))
        metrics = validate_and_plot(agent, M_val, args.max_steps, args, n_agents)
        all_metrics[beta_val] = metrics
        plt.plot()
        plt.ion()
        plt.pause(0.1)
        plt.close()
    print(all_metrics)

    # Print the metrics prettily
    for key, value in all_metrics.items():
        print(key, value)
    import pandas as pd
    df = pd.DataFrame(all_metrics)

    df_ = df.T
    #sort rows by index
    df_ = df_.sort_index()
    print(df_)

    #plot fairnes vs system_utility
    import plotly.express as px
    fig = px.scatter(df_, x="system_utility", y="fairness", color=df_.index.astype(str))
    fig.show()


## Beta region expt
all_results_region = {}
for beta in betas:
    #Evaluate the model on all other betas
    timestamp = os.listdir(folder+beta)[0]
    model_path = folder+beta+"/"+timestamp+"/best/best_model.ckpt"
    all_metrics = {}
    for beta_test in betas:
        beta_eval = float(beta_test)
        print("Evaluating beta ", beta_eval, "Base beta", beta)
        agent = load_agent(model_path, beta_eval)
        metrics = validate_and_plot(agent, M_val, args.max_steps, args, n_agents)
        all_metrics[beta_eval] = metrics
        # plt.plot()
        # plt.ion()
        # plt.pause(0.1)
        # plt.close()
    # Print the metrics prettily
    df = pd.DataFrame(all_metrics)

    df_ = df.T
    #sort rows by index
    df_ = df_.sort_index()
    print(df_)
    all_results_region[float(beta)] = all_metrics

# Print the metrics prettily
for key, value in all_results_region.items():
    print(key, value)

df = pd.DataFrame(all_results_region)
df_ = df.T
#sort rows by index
df_ = df_.sort_index()
print(df_)
#plot fairnes vs system_utility
import plotly.express as px
fig = px.scatter(df_, x="system_utility", y="fairness", color=df_.index.astype(str))