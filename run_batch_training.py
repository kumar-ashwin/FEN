import os
import json

# add argument processing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="WarmStart")
parser.add_argument("--env_name_mod", type=str, default="")
parser.add_argument("--split", type=str, default="False")
parser.add_argument("--learn_fairness", type=str, default="True")
parser.add_argument("--learn_utility", type=str, default="True")
parser.add_argument("--multi_head", type=str, default="False")
parser.add_argument("--logging", type=str, default="True")
parser.add_argument("--render", type=str, default="False")
parser.add_argument("--simple_obs", type=str, default="False")
parser.add_argument("--SI_beta", type=int, default=0)
parser.add_argument("--fairness_type", type=str, default="split_diff")
parser.add_argument("--tag", type=str, default="")
args = parser.parse_args()

params = vars(args)
#Handle boolean arguments
for k,v in params.items():
    if v == "True" or v == "False":
        params[k] = v == "True"

with open(f"hyperparams.json") as f:
    hyperparams = json.load(f)[params["env_name"]]
    for key, value in hyperparams.items():
        params[key] = value

if not params["learn_utility"]:
    if params["multi_head"]:
        print("Multihead not supported for learn_utility=False")
        exit()
    if params["simple_obs"]:
        exit()
        params["u_model_loc"] = ""
    else:
        exit()
        
learning_betas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
for learning_beta in learning_betas:
    func = f"""python training.py --learning_beta {learning_beta}"""
    for key, value in params.items():
        if value is None or value is "":
            continue
        func += f""" --{key} {value} """
    # print(func)
    os.system(func)