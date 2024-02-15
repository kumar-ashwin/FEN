@REM --model_loc "Models/JobTest/split_diff/Joint/0.0/1706820615/best/best_model.ckpt" ^
@REM Example script for testing with one agent. This should be really easy to learn utility with
python job.py ^
--training True ^
--split False ^
--learn_fairness True ^
--learn_utility True ^
--multi_head False ^
--simple_obs True ^
--logging False ^
--render False ^
--n_episode 1000 ^
--max_steps 100 ^
--learning_rate 0.0003 ^
--SI_beta 0 ^
--learning_beta 1.0 ^
--fairness_type split_diff ^
--warm_start 10