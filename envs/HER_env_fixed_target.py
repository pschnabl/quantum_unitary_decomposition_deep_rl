import numpy as np
import matplotlib.pyplot as plt
import sys
import gymnasium as gym
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit.visualization.state_visualization import _bloch_multivector_data


def get_possible_actions(gates, num_qubits):
    '''
    This function returns a dictionary of all possible actions that the agent can take in the environment.
    The keys of the dictionary are integers from 0 to num_actions-1, where num_actions is the number of possible
    actions. The values of the dictionary are dictionaries themselves, with the keys 'gate' and 'qubits'.
    The value of the key 'gate' is the gate that is applied to the qubits specified by the value of the key 'qubits'.
    '''
    actions = {}
    action_counter = 0
    for gate in gates:
        for q1 in range(num_qubits):
            if gate.num_qubits == 2:
                for q2 in range(num_qubits):
                    if q2 != q1:
                        actions[action_counter] = {'gate': gate, 'qubits': [q1, q2]}
                        action_counter += 1
            else:
                actions[action_counter] = {'gate': gate, 'qubits': [q1]}
                action_counter += 1
    return action_counter, actions

def circuit_to_unitary(circuit):
    '''
    Return the unitary matrix equivalent to the input quantum circuit.
    '''
    op = Operator(circuit)
    return op.to_matrix()

def unitary_to_goal(unitary):
    '''
    Return a representation of the unitary matrix as a vector of the real and imaginary parts of the elements
    of the matrix. This representation is used as observation.
    '''
    return np.stack([np.real(unitary),np.imag(unitary)], axis=-1).flatten()


class HEREnvFixedTarget(gym.Env):
    '''
    The environment consists of a quantum circuit that the agent can modify by applying single-
    and two-qubit gates. The goal of the agent is to approximate a fixed unitary matrix by 
    creating a corresponding efficient quantum circuit.
    '''
    def __init__(self, target_unitary, gates, num_qubits, tolerance=1e-3, max_steps=130):
        '''
        target_unitary: np.array
             the unitary matrix that the circuit should approximate
        gates: list
            gates that can be used by the agent
        num_qubits: int
            the number of qubits in the circuit
        tolerance: float
            tolerance for the average gate fidelity between the target unitary and the circuit unitary
        max_steps: int
            the maximum number of steps that the agent can take in the environment
        '''
        self.target_unitary = target_unitary
        self.stacked_target_unitary = unitary_to_goal(self.target_unitary)
        self.gates = gates
        self.num_qubits = num_qubits
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.time_step = 0
        
        self.agf = 0
        self.qc = QuantumCircuit(num_qubits)
        self.num_actions, self.action_dict = get_possible_actions(gates, num_qubits)

        # define the action space
        '''
        The action space is a discrete space where every action corresponds to placing a certain single-
        or two-qubit gate on certain qubits.
        '''
        self.action_space = gym.spaces.Discrete(self.num_actions)

        # define the observation space
        obs = self._get_obs()
        self.observation_space = gym.spaces.Dict(
            dict(
                desired_goal=gym.spaces.Box(
                    -1, 1, shape=obs["desired_goal"].shape, dtype="float64"
                ),
                achieved_goal=gym.spaces.Box(
                    -1, 1, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                observation=gym.spaces.Box(
                    -1, 1, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.qc = QuantumCircuit(self.num_qubits)
        self.agf = average_gate_fidelity(Operator(self.qc), Operator(self.target_unitary))
        self.time_step = 0
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        '''
        The step function takes an action (integer [0,num_actions-1]) as input, applies the corresponding
        gate to the corresponding qubits in the circuit, and returns the new observation, the reward, 
        and whether the episode is over.
        '''
        # get the gate and the qubits from the action
        gate = self.action_dict[action]['gate']
        qubits = self.action_dict[action]['qubits']
        # apply the selected gate to the selected qubits in the circuit
        self.qc.append(gate, qubits)

        # get the new observation
        obs = self._get_obs()
        
        # calculate the reward
        reward = self.compute_reward(obs["achieved_goal"], obs['desired_goal'], info=None)

        # check if the episode is over
        terminated = False
        truncated = False
        if (1 - self.agf) <= self.tolerance:
            terminated = True

        if self.time_step >= self.max_steps-1:
            truncated = True
        self.time_step += 1

        return obs, reward, terminated, truncated, self._get_info()
    
    def render(self, mode='human'):
        '''
        This function renders the environment.
        Possible modes: 'human', 'rgb_array', or 'statevector'.
        'human' prints the circuit in text form and the average gate fidelity.
        'rgb_array' returns an RGB array of the Bloch sphere of the circuit state.
            Useful for animating the trajectory of the agent on the Bloch sphere.
        'statevector' returns the statevector of the circuit used.
            Useful for plotting the trajectory of the agent on the Bloch sphere.
        '''
        if mode == 'human':
            outfile = sys.stdout
            outfile.write(str(self.qc.draw('text')) + '\n average gate fidelity: ' + str(self.agf))
            
        elif mode == 'rgb_array':
            # Plot the Bloch sphere
            state = Statevector(self.qc)
            fig = plot_bloch_multivector(state)

            # Convert the plot to an RGB array
            fig.canvas.draw()
            image_array = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            return image_array
        
        elif mode == 'statevector':
            state = Statevector(self.qc)
            return _bloch_multivector_data(state)
        
        else:
            raise NotImplementedError
        
    def compute_single_reward(self, achieved_goal, desired_goal):
        '''
        Computes the reward for a single goal.
        The reward is -1 if the difference between the achieved goal and the desired goal is
        not within the specified tolerance, and 0 otherwise.
        ''' 
        achieved_unitary = self.goal_to_unitary(achieved_goal)
        desired_unitary = self.goal_to_unitary(desired_goal)

        self.agf = average_gate_fidelity(Operator(achieved_unitary), Operator(desired_unitary))
        diff = 1 - self.agf
        return - float(diff >= self.tolerance)/self.max_steps

    def compute_reward(self, achieved_goals, desired_goals, info):
        '''
        Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved, which is necessary
        for HER.

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal.
        '''
        # Distinguish between calculating the reward for a single goal and a batch of goals
        # This is necessary for HER, the reward calculation needs to be vectorized.      
        if isinstance(achieved_goals[0], np.ndarray) and isinstance(desired_goals[0], np.ndarray):
            reward=[]
            for achieved_goal, desired_goal in zip(achieved_goals, desired_goals):
                reward.append(self.compute_single_reward(achieved_goal, desired_goal))
            return np.array(reward).astype(np.float32)
        else:
            return self.compute_single_reward(achieved_goals, desired_goals)
        

    def goal_to_unitary(self, goal):
        real = goal[0:-1:2].reshape(2**self.num_qubits, 2**self.num_qubits)
        imag = goal[1::2].reshape(2**self.num_qubits, 2**self.num_qubits)
        return (real + 1j*imag)

    def _get_obs(self):
        '''
        The observation at each time step corresponds to the vector of the real and imaginary parts
        of the elements of the target unitary and the circuit unitary as well as the average gate fidelity
        between them. The achieved goal is the circuit unitary, the desired goal is the target unitary.
        '''
        self.agf = average_gate_fidelity(Operator(self.qc), Operator(self.target_unitary))

        circuit_unitary = circuit_to_unitary(self.qc)
        stacked_circuit_unitary = unitary_to_goal(circuit_unitary)

        obs = np.concatenate([self.stacked_target_unitary, stacked_circuit_unitary, [self.agf]])

        achieved_goal = stacked_circuit_unitary

        desired_goal = self.stacked_target_unitary        

        return {
                "observation": obs.copy(),
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": desired_goal.copy(),
                }

    
    def _get_info(self):
        return {'AGF': self.agf}
    