import os

split = False
learn_fairness = True
learn_utility = True
multi_head = False
logging = True
render = False
simple_obs = True
n_episode = 200
max_steps = 100
SI_beta = 0
learning_rate = 0.0003
learning_beta = 1.0
fairness_type = "split_diff"
hidden_size = 20
model_update_freq = 20
best_model_update_freq = 20
model_save_freq = 20
tag=""
warm_start = 10 
if not learn_utility:
    if multi_head:
        print("Multihead not supported for learn_utility=False")
        exit()
    if simple_obs:
        u_model_loc = "Models/JobNoOccupy/Simple/split_diff/Joint/0.0/1707336611/best/best_model.ckpt"
    else:
        u_model_loc = "Models/JobNoOccupy/split_diff/Joint/0.0/1707336619/best/best_model.ckpt"
        
## 1K runs: Middle two are multihead

# learning_betas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
# learning_betas = [2.0, 5.0, 10.0, 20.0, 50.0]
learning_betas = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
learning_betas = [0, 0.05, 0.1, 0.2]
learning_betas = [0.5, 1.0, 2.0, 5.0]
learning_betas = [10.0, 20.0, 50.0, 100.0]
warm_starts = [5, 10]
# warm_starts = [1,2]
# past_discounts = [1, 0.9, 0.5]
past_discounts = [0.999, 0.995, 0.99, 0.95, 0.9]
# past_discounts = [0.9]
warm_starts = [5]
past_discounts = [0.995]

for learning_beta in learning_betas:
    for warm_start in warm_starts:
        for past_discount in past_discounts:
            newtag = tag+f"warm{warm_start}/past{past_discount}"
            os.system(f"""python warm_start.py \
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
                    {f"--tag {newtag}" if newtag else ""} \
                    --past_discount {past_discount} \
                    --warm_start {warm_start}""")