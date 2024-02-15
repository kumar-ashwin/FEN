from Agents import DDQNAgent, SplitDDQNAgent, MultiHeadDDQNAgent
import copy
from utils import get_fairness_from_su
import numpy as np
import os

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
            agent = MultiHeadDDQNAgent(M_train, num_features, hidden_size=hidden_size, learning_rate=learning_rate, replay_buffer_size=replay_buffer_size, GAMMA=GAMMA, learning_beta=learning_beta,
                    learn_utility=learn_utility, learn_fairness=learn_fairness, phased_learning=phased_training)
            if not training:
                agent.load_model(train_args.model_loc)
        else:
            agent = SplitDDQNAgent(M_train, num_features, hidden_size=hidden_size, learning_rate=learning_rate, replay_buffer_size=replay_buffer_size, GAMMA=GAMMA, learning_beta=learning_beta,
                    learn_utility=learn_utility, learn_fairness=learn_fairness)
            if not learn_utility:
                u_model_loc = train_args.u_model_loc
                agent.load_util_model(u_model_loc)
            if not learn_fairness:
                f_model_loc = train_args.f_model_loc
                agent.load_fair_model(f_model_loc)
    else:
        agent = DDQNAgent(M_train, num_features, hidden_size=hidden_size, learning_rate=learning_rate, replay_buffer_size=replay_buffer_size, GAMMA=GAMMA, learning_beta=learning_beta)
        if not training:
            agent.load_model(train_args.model_loc)

    return agent

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
            losses['FF1'].extend(loss_logs['fair'][0])
            losses['FF2'].extend(loss_logs['fair'][1])
        if train_args.learn_utility and len(loss_logs['util'])>0:
            losses['VF1'].extend(loss_logs['util'][0])
            losses['VF2'].extend(loss_logs['util'][1])
    else:
        losses1, losses2 = agent.update(num_samples=32, num_min_samples=1000)
        losses['VF1'].extend(losses1)
        losses['VF2'].extend(losses2)

    return 

def post_episode_hk(agent, losses, i_episode, train_args, args):
    # housekeeping for post-episode agent updates
    losses_dict = None
    if args.training:
        losses_dict = {k:np.mean(v) for k,v in losses.items()}
        losses_dict["Value_Loss"]= np.mean(losses["VF1"] + losses["VF2"]) 
        losses_dict["Fair_Loss"] = np.mean(losses["FF1"] + losses["FF2"])  

        if i_episode%train_args.target_update_freq == 0:
            agent.update_target_networks()
        
        #Switch the phase every phase_length episodes
        if train_args.phased_training and i_episode%train_args.phase_length==0:
            agent.switch_phase()
            print("Switched phase")
        
        # Save the model
        if i_episode%train_args.model_save_freq==0:
            os.makedirs(f"Models/{args.save_path}/", exist_ok=True)
            agent.save_model(f"Models/{args.save_path}/model_{i_episode}.ckpt")
        
    return losses_dict