from torch import nn as nn
from typing import Any, Dict
import sys

import optuna
import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gym.envs.my_envs.paper_env import PaperEnv
from rl_zoo3 import linear_schedule

import logging

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    """
    # best practices using PPO:
    # https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/best-practices-ppo.md
    # https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
    # The following hyperparamter ranges are based on the above sources

    # the number of steps taken in each environment for each update should be the same as the max_steps in the environment
    n_steps = 128 
    # the batch size should be a fraction of the buffer size (=n_steps*n_envs)
    batch_size = trial.suggest_categorical("batch_size", [int(n_steps * x) for x in [0.25, 0.5, 1]])
    
    # the discount factor determines how much weight is given to future rewards.
    gamma = trial.suggest_float("gamma", 0.7, 0.9999)

    # learning_rate corresponds to the strength of each gradient descent update step. This should typically be decreased if training is unstable, and the reward does not consistently increase.
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 3e-3, log=True)
    
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.01, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6])
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1)
    

    # other possible hyperparameters
    # n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    # net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]
    # net_arch = {
    #     "small": dict(pi=[64, 64], vf=[64, 64]),
    #     "medium": dict(pi=[256, 256], vf=[256, 256]),
    # }[net_arch]


    if batch_size > n_steps:
        batch_size = n_steps

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
    }


def optimize_agent(trial):
    """
    Train the model and optimize the hyperparameters using Optuna.
    The used objective function for the optimization is defined by training the agent for 4*10^5 timesteps,
    evaluating the policy every 10^4 timesteps and taking the negative value of the average of the rewards obtained
    by each evaluation.
    """
    model_params = sample_ppo_params(trial)
    env = PaperEnv()
    model = PPO('MlpPolicy', env, tensorboard_log='logs_hyperparamter_study3_timesteps400000', **model_params)
    timesteps = 10000
    rewards = []
    for i in range(40):
        model.learn(total_timesteps = timesteps, reset_num_timesteps = False, tb_log_name = "PPO_trial{}".format(trial.number))
        model.save("models_hyperparameter_study3_timesteps_400000/PPO_trial{}".format(trial.number))
        mean_reward, _ = evaluate_policy(model, PaperEnv(), n_eval_episodes=10)
        rewards.append(mean_reward)
    return -1 * np.mean(rewards)


if __name__ == '__main__':

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "hyperparameter_study3_timesteps_400000"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name)
    try:
        study.optimize(optimize_agent, n_trials=100, n_jobs=1)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')