import copy
import os
import json
import time

import numpy as np

from Agent_new import DoubleDQNAgent, SplitDoubleDQNAgent, MultiHeadDoubleDQNAgent
from Environment_new import WarmStartEnvt, NewJobSchedulingEnvt, MatthewEnvt, SimpleEnvt, PlantEnvt
from functools import partial
from utils import get_fairness_from_su, get_metrics_from_rewards


def get_env(env_name, args, train_args):
    env_map = {
        "WarmStart": WarmStartEnvt,
        "Job": NewJobSchedulingEnvt,
        "Matthew": partial(MatthewEnvt, GAMMA=train_args.GAMMA),
        "Simple": SimpleEnvt,
        "Plant": PlantEnvt
    }
    # Get environment specific parameters from env_config.json
    with open(f"env_config.json") as f:
        env_params = json.load(f)[env_name]
    env = env_map[env_name](**env_params, reallocate=args.reallocate, simple_obs=args.simple_obs, warm_start=args.warm_start, past_discount=args.past_discount) 
    env_train = copy.deepcopy(env)
    env_val = copy.deepcopy(env)
    return env, env_train, env_val


def get_agent(train_args, training, num_features, M_train):
    hidden_size = train_args.hidden_size
    learning_rate = train_args.learning_rate
    replay_buffer_size = train_args.replay_buffer_size
    split = train_args.split
    multi_head = train_args.multi_head
    learn_utility = train_args.learn_utility
    learn_fairness = train_args.learn_fairness
    phased_training = train_args.phased_training
    phase_length = train_args.phase_length
    
    learning_beta = train_args.learning_beta
    GAMMA = train_args.GAMMA

    if split:
        if multi_head:
            agent = MultiHeadDoubleDQNAgent(M_train, num_features, hidden_size=hidden_size, learning_rate=learning_rate, replay_buffer_size=replay_buffer_size, GAMMA=GAMMA, learning_beta=learning_beta,
                    learn_utility=learn_utility, learn_fairness=learn_fairness, phased_learning=phased_training)
            if not training:
                agent.load_model(train_args.model_loc)
        else:
            agent = SplitDoubleDQNAgent(M_train, num_features, hidden_size=hidden_size, learning_rate=learning_rate, replay_buffer_size=replay_buffer_size, GAMMA=GAMMA, learning_beta=learning_beta,
            # agent = SplitDDQNAgent(M_train, num_features, hidden_size=hidden_size, learning_rate=learning_rate, replay_buffer_size=replay_buffer_size, GAMMA=GAMMA, learning_beta=learning_beta,
                    learn_utility=learn_utility, learn_fairness=learn_fairness)
            if not learn_utility:
                u_model_loc = train_args.u_model_loc
                agent.load_util_model(u_model_loc)
            if not learn_fairness:
                f_model_loc = train_args.f_model_loc
                agent.load_fair_model(f_model_loc)
    else:
        agent = DoubleDQNAgent(M_train, num_features, hidden_size=hidden_size, learning_rate=learning_rate, replay_buffer_size=replay_buffer_size, GAMMA=GAMMA, learning_beta=learning_beta)
        # agent = DDQNAgent(M_train, num_features, hidden_size=hidden_size, learning_rate=learning_rate, replay_buffer_size=replay_buffer_size, GAMMA=GAMMA, learning_beta=learning_beta)
        if not training:
            agent.load_model(train_args.model_loc)

    return agent


def run_validation(num_epochs, M_val, agent, args, render=False, use_fair_rewards=True)->dict:
    #Run validation episodes with the current policy
    val_metrics = {'system_utility':[], 'fairness':[], 'min_utility':[], 'objective':[],'variance':[]}
    for val_eps in range(num_epochs):
        M_val.reset()
        obs = M_val.get_obs()
        score = 0
        for steps in range(args.max_steps):
            util, obs = take_env_step(M_val, agent, obs, steps, 0, args, add_to_replay=False)
            score+=util

            if render:
                M_val.render()
                time.sleep(0.1)
                print(steps, val_eps)

        print(M_val.su)

        metrics = get_metrics_from_rewards(M_val.su, args.learning_beta, fair_rewards=M_val.get_fairness_rewards())
        for key, value in metrics.items():
            val_metrics[key].append(value)
    
    return val_metrics


def save_best_model(agent, mean_val_metrics, i_episode, ep_epsilon, args, train_args):
    #make directory if it doesn't exist
    os.makedirs(f"{args.save_path}/models/best", exist_ok=True)
    agent.save_model(f"{args.save_path}/models/best/best_model.ckpt")
    print("Saved best model")
    #Write the logs to a file
    with open(f"{args.save_path}/models/best/best_log.txt", "w") as f:
        f.write(f"Validation Utility: {mean_val_metrics['system_utility']}\n")
        f.write(f"Validation Fairness: {mean_val_metrics['fairness']}\n")
        f.write(f"Validation Min Utility: {mean_val_metrics['min_utility']}\n")
        f.write(f"Validation Objective: {mean_val_metrics['objective']}\n")
        f.write(f"Validation Variance: {mean_val_metrics['variance']}\n")
        #Write the episode number, epsilon, simple_obs, reallocate, central_rewards
        f.write(f"Episode: {i_episode}\n")
        f.write(f"Epsilon: {ep_epsilon}\n")
        f.write("Arguments:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write("Training Arguments:\n")
        for arg in vars(train_args):
            f.write(f"{arg}: {getattr(train_args, arg)}\n")


def take_env_step(M, agent, obs, step, ep_epsilon, args, add_to_replay=True, use_greedy=False):
    #For each agent, select the action: central allocation
    actions = M.compute_best_actions(agent, M, obs, beta=args.SI_beta, epsilon=ep_epsilon, use_greedy=use_greedy)
    pd_states = M.get_post_decision_states(obs, actions)
    
    su_prev = copy.deepcopy(M.discounted_su)
    #Take a step based on the actions, get rewards and updated agents, resources etc.
    rewards = M.step(actions)

    util = sum(rewards)
    obs = M.get_obs()
    
    if add_to_replay:
        su_post = copy.deepcopy(M.discounted_su)
        f_rewards = get_fairness_from_su(su_prev, su_post, ftype=args.fairness_type, action=actions)

        #Add to replay buffer
        #Experience stores - [(s,a), r(s,a,s'), r_f(s,a,s'), s']
        done = step==args.max_steps
        agent.add_experience(copy.deepcopy(pd_states), copy.deepcopy(rewards), copy.deepcopy(f_rewards), copy.deepcopy(M.get_state()), done)
    return util, obs


def training_step(agent, train_args, losses, num_samples=32, num_min_samples=1000):
    if train_args.split:
        loss_logs = agent.update(num_samples=num_samples, num_min_samples=num_min_samples)
        if train_args.learn_fairness and len(loss_logs['fair'])>0:
            losses['FF'].extend(loss_logs['fair'])
        if train_args.learn_utility and len(loss_logs['util'])>0:
            losses['VF'].extend(loss_logs['util'])
    else:
        step_losses = agent.update(num_samples=num_samples, num_min_samples=num_min_samples)
        losses['VF'].extend(step_losses)

    return 


def post_episode_hk(agent, losses, i_episode, train_args, args):
    # housekeeping for post-episode agent updates
    losses_dict = None
    if args.training:
        losses_dict = {k:np.mean(v) for k,v in losses.items()}
        losses_dict["Value_Loss"]= np.mean(losses["VF"]) 
        losses_dict["Fair_Loss"] = np.mean(losses["FF"])  

        if i_episode%train_args.target_update_freq == 0:
            agent.update_target_network()
        
        #Switch the phase every phase_length episodes
        if train_args.phased_training and i_episode%train_args.phase_length==0:
            agent.switch_phase()
            print("Switched phase")
        
        # Save the model
        if i_episode%train_args.model_save_freq==0:
            os.makedirs(f"{args.save_path}/models/", exist_ok=True)
            agent.save_model(f"{args.save_path}/models/model_{i_episode}.ckpt")
        
    return losses_dict