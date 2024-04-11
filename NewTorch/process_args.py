from dataclasses import dataclass, field
from time import time
from typing import List

import transformers
import os

import configparser

@dataclass
class RunArguments:
    greedy: bool = field(
        default=False,
        metadata={"help": "Whether to use greedy action selection or not"},
    )
    training: bool = field(
        default=True,
        metadata={"help": "Whether to train the model or not"},
    )
    reallocate: bool = field(
        default=False,
        metadata={"help": "Whether to reallocate rewards or not"},
    )
    central_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to use central rewards or not"},
    )
    simple_obs: bool = field(
        default=False,
        metadata={"help": "Whether to use simple observations or not"},
    )
    logging: bool = field(
        default=True,
        metadata={"help": "Logs to tensorboard if True"},
    )
    env_name: str = field(
        default="",
        metadata={"help": "Name of the environment to use. Used for logging, and for setting up the environment from env_config.json"},
    )
    env_name_mod: str = field(
        default="",
        metadata={"help": "Modifier for saving results."},
    )

    # Fairness parameters
    SI_beta: int = field(
        default=0,
        metadata={"help": "Beta parameter for SI fairness"},
    )
    learning_beta: float = field(
        default=0.0,
        metadata={"help": "Beta parameter for fairness learning"},
    )
    fairness_type: str = field(
        default="split_diff",
        metadata={"help": "Type of fairness to use. Options are 'split_diff', 'variance_diff', 'split_variance', 'variance', 'SI'"},
    )

    # Mode
    tag: str = field(
        default="",
        metadata={"help": "Tag for the run. Used for logging"},
    )

    warm_start: float = field(
        default=50.0,
        metadata={"help": "Warm start value for fairness"},
    )
    past_discount: float = field(
        default=0.995,
        metadata={"help": "Discount factor for past rewards, to reduce importance of old rewards"},
    )


    # Training parameters
    n_episode: int = field(
        default=1000,
        metadata={"help": "Number of episodes to train for"},
    )
    max_steps: int = field(
        default=100,
        metadata={"help": "Maximum number of steps in each episode"},
    )
    render: bool = field(
        default=False,
        metadata={"help": "Whether to render the environment or not"},
    )



@dataclass
class TrainingArguments():
    hidden_size: int = field(default=20)
    learning_rate: float = field(default=0.003)
    replay_buffer_size: int = field(default=250000)
    model_update_freq: int = field(
        default=50,
        metadata={"help": "Number of steps between model updates"},
    )
    target_update_freq: int = field(
        default=20,
        metadata={"help": "Number of episodes between target model updates"},
    )
    model_save_freq: int = field(
        default=100,
        metadata={"help": "Number of episodes between model saves"},
    )
    validation_freq: int = field(
        default=10,
        metadata={"help": "Number of episodes between validation runs"},
    )
    best_model_update_freq: int = field(
        default=100,
        metadata={"help": "Number of episodes between checking if the current model is the best on validation set"},
    )
    GAMMA: float = field(
        default=0.98,
        metadata={"help": "Discount factor for future rewards"},
    )

    model_loc: str = field(
        default="",
        metadata={"help": "Location of the model to load"},
    )
    u_model_loc: str = field(
        default="",
        metadata={"help": "Location of the utility model to load"},
    )
    f_model_loc: str = field(
        default="",
        metadata={"help": "Location of the fairness model to load"},
    )

    #Type of model
    split: bool = field(
        default=False,
        metadata={"help": "Use separate nets for fairness and utility"},
    )
    learn_fairness: bool = field(
        default=True,
        metadata={"help": "Whether to learn fairness or not"},
    )
    learn_utility: bool = field(
        default=True,
        metadata={"help": "Whether to learn utility or not"},
    )
    multi_head: bool = field(
        default=False,
        metadata={"help": "Whether to use multi-head model or not"},
    )

    phased_training: bool = field(
        default=False,
        metadata={"help": "Phased training cycles between fairness and utility"},
    )
    phase_length: int = field(
        default=200,
        metadata={"help": "Length of each phase in phased training"},
    )



def process_args(env_name=None, config_file=None, load_default=False):
    if load_default:
        run_args = RunArguments()
        training_args = TrainingArguments()
    elif config_file is not None and os.path.isfile(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        # Parse arguments from config file
        run_args = RunArguments()
        training_args = TrainingArguments()

        if 'RunArguments' in config:
            for key, value in config['RunArguments'].items():
                setattr(run_args, key, value)

        if 'TrainingArguments' in config:
            for key, value in config['TrainingArguments'].items():
                setattr(training_args, key, value)
    else:
        # Parse arguments from command line
        parser = transformers.HfArgumentParser((RunArguments, TrainingArguments))
        run_args, training_args = parser.parse_args_into_dataclasses()

    if env_name is not None:
        run_args.env_name = env_name
    else:
        assert run_args.env_name != "", "Environment name not provided"
    # Process run arguments
    st_time = int(time())

    # Set save path
    mode = "Reallocate" if run_args.reallocate else ""
    mode += "Central" if run_args.central_rewards else ""
    mode += "Simple" if run_args.simple_obs else ""
    
    mode += f"/{run_args.fairness_type}"

    network_type = ""
    if training_args.multi_head:
        network_type = "/MultiHead"
    elif training_args.split and not training_args.multi_head:
        network_type = "/Split"
    elif not training_args.split:
        network_type = "/Joint"
    mode+= network_type
    mode += "Phased" if training_args.phased_training else ""
    if training_args.split and not training_args.learn_utility:
        mode += "NoUtility"
    if training_args.split and not training_args.learn_fairness:
        mode += "NoFairness"
    # mode += "/"+run_args.tag
    mode += f"/{run_args.learning_beta}"

    mode = run_args.env_name + run_args.env_name_mod + "/" + run_args.tag + "/" + mode

    # Check the save path and list the folders in the directory
    save_path = "logs/"+mode
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    exp_count = len(os.listdir(save_path))
    num_bootstraps = 5
    if len(os.listdir(save_path)) >= num_bootstraps:
        print("Too many experiments in this folder. Exiting")
        #TODO: Write a script to check completed experiments and purge the folder
        exit()
    exp_num = exp_count+1
    for i in range(1, exp_count+1):
        if str(i) not in os.listdir(save_path):
            exp_num = i
            break
    save_path_exp = save_path + f"/{exp_num}"
    os.mkdir(save_path_exp)
    print(f"Save path: {save_path}")
    # exit()
    run_args.save_path = save_path_exp

    training_args.learning_beta = run_args.learning_beta

    return run_args, training_args