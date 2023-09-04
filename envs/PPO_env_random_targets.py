import numpy as np
import matplotlib.pyplot as plt
import sys

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, average_gate_fidelity, random_unitary
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit.visualization.state_visualization import _bloch_multivector_data

import gymnasium as gym


def get_possible_actions(gates, num_qubits):
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
    op = Operator(circuit)
    return op.to_matrix()


class PPOEnvRandomTargets(gym.Env):
    '''
    The environment consists of a quantum circuit that can be modyfied by the agent by applying single-
    and two-qubit gates. The goal of the agent is to approximate arbitrary unitary matrices by 
    creating a corresponding efficient quantum circuit.
    '''
    def __init__(self, gates, num_qubits, tolerance=1e-3, max_steps=130):
        '''
        In this environment the target unitary is chosen randomly each episode.

        gates: list
            gates that can be used by the agent
        num_qubits: int
            the number of qubits in the circuit
        tolerance: float
            tolerance for the average gate fidelity between the target unitary and the circuit unitary
        max_steps: int
            the maximum number of steps that the agent can take in the environment
        '''
        self.num_qubits = num_qubits
        self.gates = gates
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.time_step = 0

        self.qc = QuantumCircuit(num_qubits)
        self.num_actions, self.action_dict = get_possible_actions(gates, num_qubits)
        self.agf = 0

        self.target_unitary = random_unitary(2**self.num_qubits, seed=self.np_random).to_matrix()

        # define the observation space
        '''
        The observation at each time step corresponds to the vector of the real and imaginary parts 
        of the elements of the matrix On, where U_target = U_circuit ⋅ On
        '''
        obserbvation_shape = np.stack([np.real(self.target_unitary), np.imag(self.target_unitary)], axis=-1).flatten().shape
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=obserbvation_shape, dtype='float64')

       # define the action space
        '''
        The action space is a discrete space where every action corresponds to placing a certain single-
        or two-qubit gate on certain qubits.
        '''
        self.action_space = gym.spaces.Discrete(self.num_actions)

    def reset(self,seed=None, options=None):
        '''	
        The reset function resets the environment to its initial state and returns the initial observation
        and info.
        '''
        super().reset(seed=seed)
        self.target_unitary = random_unitary(2**self.num_qubits, seed=self.np_random).to_matrix()
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
        gate = self.action_dict[action]['gate']
        qubits = self.action_dict[action]['qubits']
        self.qc.append(gate, qubits)

        # calculate the difference between the target unitary and the circuit unitary using the average gate fidelity
        self.agf = average_gate_fidelity(Operator(self.qc), Operator(self.target_unitary))
        diff = 1 - self.agf

        # calculate the reward
        L = self.max_steps
        if diff < self.tolerance:
            reward = (L - self.time_step) + 1
        else:
            reward = -diff / L

        # Terminate the episode if the agent has taken the maximum number of steps
        done = False
        terminated = False
        if self.time_step >= self.max_steps-1:
            terminated = True

        self.time_step += 1
        return self._get_obs(), reward, terminated, done, self._get_info()
    
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
            outfile = sys.stdout # StringIO() if mode == 'ansi' else sys.stdout
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

        
    def _get_obs(self):
        '''
        The observation at each time step corresponds to the vector of the real and imaginary parts 
        of the elements of the matrix O, where U_target = U_circuit ⋅ On
        '''
        circuit_unitary = circuit_to_unitary(self.qc)
        O = np.linalg.inv(circuit_unitary) @ self.target_unitary
        return np.stack([np.real(O), np.imag(O)], axis=-1).flatten()
    
    def _get_info(self):
        return {'AGF': self.agf}
    