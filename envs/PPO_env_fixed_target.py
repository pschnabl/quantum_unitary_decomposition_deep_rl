import numpy as np
import matplotlib.pyplot as plt
import sys

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit.visualization.state_visualization import _bloch_multivector_data

import gym


def get_possible_actions(gates, num_qubits):
    '''
    Returns a dictionary of all possible actions that the agent can take in the environment.
    The keys of the dictionary are integers from 0 to num_actions-1, where num_actions is the
    number of possible actions. The values of the dictionary are dictionaries themselves, with
    the keys 'gate' and 'qubits'. The value of the key 'gate' is the gate that is applied to the
    qubits specified by the value of the key 'qubits'.
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
    op = Operator(circuit)
    return op.to_matrix()


class PPOEnvFixedTarget(gym.Env):
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
        self.gates = gates
        self.num_qubits = num_qubits
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.time_step = 0
        

        self.qc = QuantumCircuit(num_qubits)
        self.num_actions, self.action_dict = get_possible_actions(gates, num_qubits)
        self.agf = 0

        # define the observation space
        '''
        The observation at each time step corresponds to the vector of the real and imaginary parts 
        of the elements of the matrix On, where U_target = U_circuit ⋅ On
        '''
        obserbvation_shape = np.stack([np.real(target_unitary), np.imag(target_unitary)], axis=-1).flatten().shape
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=obserbvation_shape, dtype='float64')

       # define the action space
        '''
        The action space is a discrete space where every action corresponds to placing a certain single-
        or two-qubit gate on certain qubits.
        '''
        self.action_space = gym.spaces.Discrete(self.num_actions)

    def reset(self):
        self.qc = QuantumCircuit(self.num_qubits)
        self.agf = average_gate_fidelity(Operator(self.qc), Operator(self.target_unitary))
        self.time_step = 0
        return self._get_obs()
    
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

        # calculate the dense reward
        L = self.max_steps
        if diff < self.tolerance:
            reward = (L - self.time_step) + 1
        else:
            reward = -diff / L

        # The episode is over if the agent has taken the maximum number of steps
        done = False
        if self.time_step >= self.max_steps-1:
            done = True
        self.time_step += 1

        return self._get_obs(), reward, done, {'AGF': self.agf}
    
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
        
    def _get_obs(self):
        '''
        The observation at each time step corresponds to the vector of the real and imaginary parts 
        of the elements of the matrix O, where U_target = U_circuit ⋅ On
        '''
        circuit_unitary = circuit_to_unitary(self.qc)
        O = np.linalg.inv(circuit_unitary) @ self.target_unitary
        return np.stack([np.real(O), np.imag(O)], axis=-1).flatten()
    