import os

split = False
learn_fairness = True
learn_utility = True
multi_head = False
logging = True
render = False
n_episode = 1000
max_steps = 100
SI_beta = 0
learning_beta = 1.0
fairness_type = "split_diff"
warm_start = 10


learning_betas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
learning_betas = [2.0, 5.0, 10.0, 20.0, 50.0]
for learning_beta in learning_betas:
    os.system(f"""python job.py \
            --split {split} \
            --learn_fairness {learn_fairness} \
            --learn_utility {learn_utility} \
            --multi_head {multi_head} \
            --logging {logging} \
            --render {render} \
            --n_episode {n_episode} \
            --max_steps {max_steps} \
            --SI_beta {SI_beta} \
            --learning_beta {learning_beta} \
            --fairness_type {fairness_type} \
            --warm_start {warm_start}""")