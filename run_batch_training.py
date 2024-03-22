import os
import json

params = {
    "split": False,
    "learn_fairness": True,
    "learn_utility": True,
    "multi_head": False,
    "logging": True,
    "render": False,
    "simple_obs": False,
    "SI_beta": 0,
    "fairness_type": "split_diff",
    "tag": "",
    "env_name": "Plant",
    "env_name_mod": ""
}

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
        
learning_betas = [0.0]#, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
for learning_beta in learning_betas:
    func = f"""python training.py --learning_beta {learning_beta}"""
    for key, value in params.items():
        if value is None or value is "":
            continue
        func += f""" --{key} {value} """
    print(func)
    os.system(func)