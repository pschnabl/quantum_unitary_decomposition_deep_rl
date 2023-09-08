# Deep Reinforcement Learning for Quantum Unitary Matrix Decomposition

This repository hosts the source code accompanying the master's thesis by Paul Schnabl, which can be accessed [here](). The research focuses on the application of deep reinforcement learning techniques to decompose single-qubit unitary matrices into sequences of quantum gates, chosen from a predefined set of basis gates. The study aims to address two primary challenges:

1. Training a reinforcement learning agent to find an efficient decomposition for a specific *fixed* target unitary matrix.
2. Developing an agent capable of generating decompositions for arbitrary single-qubit unitary matrices.

The aforementioned challenges are tackled using state-of-the-art reinforcement learning algorithms, namely [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347), [Deep Q-Networks (DQN)](https://arxiv.org/abs/1312.5602), and [DQN combined with Hindsight Experience Replay (DQN+HER)](https://arxiv.org/abs/1707.01495).


## Dependencies and Libraries
The implementation is executed in Python and relies on the following key libraries:

* [Gymnasium](https://gymnasium.farama.org/) - Utilized for constructing customized reinforcement learning environments that encapsulate the unitary matrix decomposition problem.
* [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) - Employed for training the reinforcement learning agents. This library provides implementations for the PPO and DQN/DQN+HER algorithms.
* [Qiskit](https://qiskit.org/) - Leveraged for simulating the quantum circuits that the agents are designed to generate.


## Usage
### Training
Agents can be trained using the "train*.py" scripts and the corresponding costum Gymnasium environments in the envs folder. The training progress of the agent can be visualized using Tensorboard. The following command can be used to launch Tensorboard:

```bash tensorboard --logdir ./logs/```

During the training models of the trained agents policies can be stored and loaded later on for evaluation.

### Evaluation
The "evaluate*.py" scripts can be used to load a trained agent and evaluate the performance. The scripts contain three kinds of visualization:
* Circuit visualization: The generated quantum circuit is visualized using Qiskit.
        ![Circuit visualization](./imgs/circuit_visualization.mp4)
* Bloch sphere animation: The evolution of the quantum state on the Bloch sphere is visualized.
* Bloch sphere trajectory: The trajectory of the quantum state on the Bloch sphere is visualized.
