from torch import nn as nn
from typing import Any, Dict
import sys

import optuna
import gymnasium as gym
import numpy as np
import pandas as pd

from HER_env import HEREnv
from qiskit.circuit.library.standard_gates import RZGate, RXGate, RYGate, SXGate, XGate

from stable_baselines3 import DQN, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import logging


def get_reward_dataframe(path):
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    reward_scalars = event_acc.Scalars('rollout/ep_rew_mean')
    df = pd.DataFrame()

    steps = [reward.step for reward in reward_scalars]
    data = [reward.value for reward in reward_scalars]
    df['steps'] = pd.Series(steps)
    df['reward'] = pd.Series(data)
    return df

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # log the average gate fidelity
        self.logger.record("average gate fidelity", self.training_env.buf_infos[-1]['AGF'])
        return True

def sample_DQNHER_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DQN + HER hyperparams.
    Optuna samples the specified hyperparamters from the given ranges or list of values.
    """

    # maximum number of steps taken in the environment per episode (this is a hyperparamter of our specific environment not the DQN algorithm)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 320])

    # the batch size should be smaller or equal to the number of steps taken in the environment per episode
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    if batch_size > n_steps:
        batch_size = n_steps

    # the discount factor determines how much weight is given to future rewards.
    gamma = trial.suggest_categorical("gamma", [0.99, 0.999, 0.9999])

    # learning_rate corresponds to the strength of each update step. This should typically be decreased if training is unstable, and the reward does not consistently increase.
    learning_rate = trial.suggest_categorical("learning_rate", [0.0001, 0.00008, 0.00005, 0.00002])

    # Parameters for Hindisght Experience Replay (HER)
    replay_buffer_kwargs = dict(
        # Available strategies (cf paper): future, final, episode
        goal_selection_strategy = trial.suggest_categorical("goal_selection_strategy", ["future", "final", "episode"]),
        n_sampled_goal = trial.suggest_categorical("n_sampled_goal", [4, 8, 16])
    )

    # network architecture
    net_arch = trial.suggest_categorical("net_arch", ["small", "large"])
    net_arch = {"small": [64, 64], "large": [256, 256]}[net_arch]

    # exploration fraction determines the fraction of the total number of steps taken in the environment where the epsilon value is linearly annealed from 1 to exploration_final_eps
    exploration_fraction= trial.suggest_categorical("exploration_fraction", [0.1, 0.3, 0.5])

    # train_freq determines how often the model is updated.
    train_freq=trial.suggest_categorical("train_freq", [1, 2, 4]) # Update the model every n episodes
    train_freq = train_freq * n_steps

    # seed for the random number generator, to make experiments reproducible
    seed = trial.suggest_categorical("seed", [0, 42, 1234, 88, 8888, 1000])

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "replay_buffer_kwargs": replay_buffer_kwargs,
        "exploration_fraction": exploration_fraction,
        "train_freq": train_freq,
        "seed": seed,
        "policy_kwargs": dict(net_arch=net_arch)
    }


def optimize_agent(trial):
    """
    Train the model and optimize the hyperparameters using Optuna.
    The mean negative reward over the specified number of training steps is used
    as a metric to optimize the hyperparameters.
    """
    # define number of qubits of the quantumm circuit
    model_params = sample_DQNHER_params(trial)

    NUM_QUBITS = 1
    TOLERANCE = 1e-2

    MAX_STEPS = model_params["n_steps"]

    # delete the n_steps parameter from the model_params dictionary, since it is not a parameter of the DQN algorithm but of our environment
    model_params.pop("n_steps")

    ibm_single_qubit_gate_set = [SXGate(), XGate(), RZGate(np.pi/16), RZGate(-np.pi/16)]
    GATES = ibm_single_qubit_gate_set

    # Target unitary used in the paper "Quantum Compiling by Deep Reinforcement Learning" https://www.nature.com/articles/s42005-021-00684-3
    U = [[0.76749896-0.43959894j, -0.09607122+0.45658344j],[0.09607122+0.45658344j, 0.76749896+0.43959894j]]

    # define the environment
    env = HEREnv(U, GATES, NUM_QUBITS, tolerance=TOLERANCE, max_steps=MAX_STEPS)

    # define the directory where the models and logs are saved
    models_dir = 'hyperparameter_study_models/' + "trial{}".format(trial.number) # trial.number is the id of the current optuna trial
    logdir = 'hyperparameter_study_logs/' + "trial{}".format(trial.number)

    # Initialize the model
    model_class = DQN 
    model = model_class(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        verbose=1,
        tensorboard_log=logdir,
        **model_params  # pass the hyperparameters suggested by optuna to the model
    )

    # create a separate evaluation env in order to save the best model
    eval_env = HEREnv(U, GATES, NUM_QUBITS, tolerance=TOLERANCE, max_steps=MAX_STEPS)
    # define the callback during training
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{models_dir}/best_model",
                                log_path="./eval_logs/results", eval_freq=10000)

    # Create the callback list
    callback = CallbackList([TensorboardCallback(), eval_callback])

    # Define the number of timesteps to train for and on which the performance of the training is evaluated
    TIMESTEPS = int(2e6)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=callback)

    # get the mean reward over the total number of episodes
    reward_df = get_reward_dataframe(logdir + '/DQN_0') # the "DQN_0" directory is created by default
    mean_reward = reward_df['reward'].mean()
    print("Mean reward:", mean_reward)
    return -1 * mean_reward


if __name__ == '__main__':

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "hyperparameter_study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name) 
    
    # create a database to store the results of the hyperparameter optimization, which can be accessed later using optuna-dashboard
    study = optuna.create_study(study_name=study_name, storage=storage_name)
    try:
        study.optimize(optimize_agent, n_trials=100, n_jobs=1) # train the model for 100 trials
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')