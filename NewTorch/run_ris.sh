# #!/bin/bash
env_name="WarmStart"
env_name_mod=""
tag=""
u_model_loc=""
# split="False"
# learn_utility="True"
# multi_head="False"
# logging="True"
# render="False"
# SI_beta=0
# fairness_type="split_diff"
learning_betas="0.0 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0 50.0"
if [ "$env_name" = "WarmStart" ]; then
    learning_betas="0 1 10 100 500 1000 5000 10000 50000 100000"
fi

# Add all parameters as --param value
# split=${split} \
# learn_utility=${learn_utility} \
# multi_head=${multi_head} \

func="run_training_ris.py "
params_dict="\
env_name=${env_name} \
env_name_mod=${env_name_mod} \
tag=${tag} \
u_model_loc=${u_model_loc}"

for key in $params_dict
do
    value=$(echo $key | cut -d'=' -f2)
    key=$(echo $key | cut -d'=' -f1)
    if [ "$value" = "" ]; then
        continue
    fi
    func="${func} --${key} ${value}"
done

echo $func

# for env in "WarmStart"
# do
for learning_beta in $learning_betas
do

    declare -A dict
    dict["joint"]="True False False"
    # dict["only_fairness"]="False True False"
    dict["multi_head"]="True True True"
    dict["split"]="True True False"

    conda_loc="/storage1/fs1/wyeoh/Active/ashwin/.conda/envs/fen/bin/python"

    for key in "${!dict[@]}"
    do
        echo $env_name $key $learning_beta
        # echo ${dict[$key]}
        learn_utility=$(echo ${dict[$key]} | cut -d' ' -f1)
        split=$(echo ${dict[$key]} | cut -d' ' -f2)
        multi_head=$(echo ${dict[$key]} | cut -d' ' -f3)

        echo "${conda_loc} ${func} --learning_beta ${learning_beta} --learn_utility ${learn_utility} --split ${split} --multi_head ${multi_head}"

        bsub -n 1 \
        -q general \
        -m general \
        -G compute-wyeoh \
        -J ${env}_${split}_${multi_head}_${learn_utility} \
        -M 4GB \
        -N \
        -u ashwinkumar@wustl.edu \
        -o /storage1/fs1/wyeoh/Active/ashwin/job_output/FEN/${env}_${split}_${multi_head}_${learn_utility}.%J.txt \
        -R 'rusage[mem=4GB] span[hosts=1]' \
        -g /ashwinkumar/limit100 \
        -a "docker(rapidsai/rapidsai:21.10-cuda11.0-runtime-ubuntu20.04-py3.8)" \
        "cd ~/Git/FEN/NewTorch/ && ${conda_loc} ${func} --learning_beta ${learning_beta} --learn_utility ${learn_utility} --split ${split} --multi_head ${multi_head}"
    done
done
