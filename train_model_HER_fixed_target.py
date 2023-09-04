import numpy as np
from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate, SXGate, XGate
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3 import HerReplayBuffer, DQN
from envs.HER_env_fixed_target import HEREnvFixedTarget
import os


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

# define the gates that can be used by the agent to decompose the target unitary
# IBM Single Qubit Gate Set
ibm_single_qubit_gate_set = [SXGate(), XGate(), RZGate(-np.pi/16) ,RZGate(np.pi/16)]

# Rotation Gate Set
rotation_gates = [RXGate(np.pi/128), RXGate(-np.pi/128),
                  RYGate(np.pi/128), RYGate(-np.pi/128),
                  RZGate(np.pi/128), RZGate(-np.pi/128)]

GATES = ibm_single_qubit_gate_set

# Define the Target Unitary (this is the same unitary as in the paper "Quantum compiling by deep reinforcement learning " )
U = [[0.76749896-0.43959894j, -0.09607122+0.45658344j],
     [0.09607122+0.45658344j, 0.76749896+0.43959894j]]

env = HEREnvFixedTarget(target_unitary=U, gates=GATES, num_qubits=NUM_QUBITS, tolerance=TOLERANCE, max_steps=MAX_STEPS)

# define the model, we use DQN in combination with Hindsight Experience Replay (HER)
model_class = DQN

# Select the goal selection strategy for the virtual goals in HER
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

models_dir = 'models/'
logdir = 'logs/'

# Initialize the model and set the hyperparameters
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,    # Here we initiate using HER
    replay_buffer_kwargs=dict(              # Keyword arguments for the replay buffer.
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        ),
    learning_rate=0.00008,                  # changing the learning rate from 0.0003 to 0.00008
    verbose=1,
    tensorboard_log=logdir,
    seed=99                                 # set the seed for reproducibility of the training
)

# create a separate evaluation env in order to save the best model
eval_envs = HEREnvFixedTarget(U, GATES, NUM_QUBITS, TOLERANCE, MAX_STEPS)

# define the callback during training
eval_callback = EvalCallback(eval_envs, best_model_save_path=f"{models_dir}/best_model",
                             log_path="./eval_logs/results", eval_freq=10000)

# Create the callback list to log the training to tensorboard and save the models
callback = CallbackList([TensorboardCallback(), eval_callback, TrainAndLoggingCallback(10000, models_dir)])

# Train the model
TIMESTEPS = int(5e6)
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="HER", callback=callback)

# save the final model aswell
model.save(f"{models_dir}/final_model")