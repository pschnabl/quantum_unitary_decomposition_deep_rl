import numpy as np
import os
from qiskit.circuit.library.standard_gates import RZGate, RXGate, RYGate, SXGate, XGate
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from torch import nn as nn
from envs.PPO_env_fixed_target import PPOEnvFixedTarget


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # log the mean average gate fidelity of all environments in the vectorized environment
        self.logger.record("mean average gate fidelity", np.mean([agf['AGF'] for agf in self.training_env.buf_infos]))
        return True
    
class TrainAndLoggingCallback(BaseCallback):
    """
    Costum callback for saving the model every check_freq steps
    """
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

# define number of qubits of the quantumm circuit
NUM_QUBITS = 1
TOLERANCE = 1e-2
MAX_STEPS = 128
NUM_ENVS = 4

# define the gates that can be used by the agent to decompose the target unitary
# IBM Single Qubit Gate Set
ibm_single_qubit_gate_set = [SXGate(), XGate(), RZGate(-np.pi/16) ,RZGate(np.pi/16)]

# Rotation Gate Set
rotation_gates = [RXGate(np.pi/128), RXGate(-np.pi/128),
                  RYGate(np.pi/128), RYGate(-np.pi/128),
                  RZGate(np.pi/128), RZGate(-np.pi/128)]

GATES = rotation_gates

# Define the Target Unitary (this is the same unitary as in the paper "Quantum compiling by deep reinforcement learning " )
U = [[0.76749896-0.43959894j, -0.09607122+0.45658344j],
     [0.09607122+0.45658344j, 0.76749896+0.43959894j]]

env = PPOEnvFixedTarget(target_unitary=U, gates=GATES, num_qubits=NUM_QUBITS, tolerance=TOLERANCE, max_steps=MAX_STEPS)

models_dir = 'models/'
logdir = 'logs/'

envs = make_vec_env(lambda: PPOEnvFixedTarget(U, GATES, NUM_QUBITS, TOLERANCE, MAX_STEPS), NUM_ENVS)
model = PPO('MlpPolicy', envs, n_steps=MAX_STEPS, verbose=1, tensorboard_log=logdir, seed=100)

# create a separate evaluation env in order to save the best model
eval_envs = make_vec_env(lambda: PPOEnvFixedTarget(U, GATES, NUM_QUBITS, TOLERANCE, MAX_STEPS), NUM_ENVS)

# define the callback during training
eval_callback = EvalCallback(eval_envs, best_model_save_path=f"{models_dir}/best_model",
                             log_path="./eval_logs/results", deterministic=True, eval_freq=10000)

# Create the callback list to log the training to tensorboard and save the models
callback = CallbackList([TensorboardCallback(), eval_callback, TrainAndLoggingCallback(10000, models_dir)])

# Train the agent
TIMESTEPS = int(2e6)
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", callback=callback)
