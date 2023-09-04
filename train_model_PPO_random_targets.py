import numpy as np
import os
from qiskit.circuit.library.standard_gates import RZGate, RXGate, RYGate, SXGate, XGate
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from torch import nn as nn
from envs.PPO_env_random_targets import PPOEnvRandomTarget


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
MAX_STEPS = 320
NUM_ENVS = 40

# define the gates that can be used by the agent to decompose the target unitary
# IBM Single Qubit Gate Set
ibm_single_qubit_gate_set = [SXGate(), XGate(), RZGate(-np.pi/16) ,RZGate(np.pi/16)]

# Rotation Gate Set
rotation_gates = [RXGate(np.pi/128), RXGate(-np.pi/128),
                  RYGate(np.pi/128), RYGate(-np.pi/128),
                  RZGate(np.pi/128), RZGate(-np.pi/128)]

GATES = rotation_gates


env = PPOEnvRandomTarget(gates=GATES, num_qubits=NUM_QUBITS, tolerance=TOLERANCE, max_steps=MAX_STEPS)


models_dir = 'models/'
logdir = 'logs/'

# use a policy with two hidden layers of 128 neurons each
policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=nn.SELU)

envs = make_vec_env(lambda: PPOEnvRandomTarget(gates=GATES, num_qubits=NUM_QUBITS, tolerance=TOLERANCE, max_steps=MAX_STEPS), NUM_ENVS)
model = PPO('MlpPolicy', envs, n_steps=MAX_STEPS, verbose=1, batch_size=128, learning_rate=0.0001, seed=0, policy_kwargs=policy_kwargs, tensorboard_log=logdir)

# create a separate evaluation env in order to save the best model
eval_envs = make_vec_env(lambda: PPOEnvRandomTarget(gates=GATES, num_qubits=NUM_QUBITS, tolerance=TOLERANCE, max_steps=MAX_STEPS), NUM_ENVS)

# define the callback during training
eval_callback = EvalCallback(eval_envs, best_model_save_path=f"{models_dir}/best_model",
                             log_path="./eval_logs/results", eval_freq=10000)

# Create the callback list to log the training to tensorboard and save the models
callback = CallbackList([TensorboardCallback(), eval_callback, TrainAndLoggingCallback(10000, models_dir)])

# train the model for 100 million timesteps
TIMESTEPS = int(100e6)
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", callback=callback)