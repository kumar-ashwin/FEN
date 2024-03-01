import os

split = False
learn_fairness = True
learn_utility = True
multi_head = False
logging = True
render = False
simple_obs = False
n_episode = 1000
max_steps = 100
SI_beta = 0
learning_rate = 0.0003
fairness_type = "split_diff"
hidden_size = 256
model_update_freq = 50
best_model_update_freq = 100
model_save_freq = 100
tag=""
warm_start = 3
if not learn_utility:
    if multi_head:
        print("Multihead not supported for learn_utility=False")
        exit()
    if simple_obs:
        exit()
        u_model_loc = ""
    else:
        exit()
        
learning_betas = [0.0]#, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
for learning_beta in learning_betas:
    os.system(f"""python plant.py \
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
            --model_save_freq {model_save_freq} \
            --best_model_update_freq {best_model_update_freq} \
            {f"--u_model_loc {u_model_loc}" if not learn_utility else ""} \
            {f"--tag {tag}" if tag else ""} \
            --warm_start {warm_start}""")