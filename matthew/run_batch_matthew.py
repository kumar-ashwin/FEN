import os

split = True
learn_fairness = True
learn_utility = False
multi_head = False
logging = True
render = False
simple_obs = False
n_episode = 1500
max_steps = 500
SI_beta = 0
learning_rate = 0.0003
learning_beta = 1.0
fairness_type = "split_diff"
model_update_freq = 100
hidden_size = 20 # Use 256?
tag="Main1500"
warm_start = 50 
if not learn_utility:
    if multi_head:
        print("Multihead not supported for learn_utility=False")
        exit()
    if simple_obs:
        exit()
        u_model_loc = ""
    else:
        u_model_loc = "Models/Matthew/Main1500/split_diff/Joint/0.0/1708110454/best/best_model.ckpt"
        # exit()
        
## All models so far were with default hidden size (20)
    

# learning_betas = [0.0, 0.01, 0.02, 0.05]
learning_betas = [0.01, 0.02, 0.05, 0.1, 0.2] #Skipping 0 for no util, just for now
#50 is incomplete
learning_betas = [0.1, 0.2, 0.3, 0.4, 0.5]
# learning_betas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
for learning_beta in learning_betas:
    os.system(f"""python matthew.py \
            --split {split} \
            --learn_fairness {learn_fairness} \
            --learn_utility {learn_utility} \
            --multi_head {multi_head} \
            --hidden_size {hidden_size} \
            --logging {logging} \
            --render {render} \
            --simple_obs {simple_obs} \
            --n_episode {n_episode} \
            --max_steps {max_steps} \
            --SI_beta {SI_beta} \
            --learning_beta {learning_beta} \
            --fairness_type {fairness_type} \
            --learning_rate {learning_rate} \
            --model_update_freq {model_update_freq} \
            {f"--u_model_loc {u_model_loc}" if not learn_utility else ""} \
            {f"--tag {tag}" if tag else ""} \
            --warm_start {warm_start}""")