
import numpy as np

# Torque / Continiuos: DDPG, TD3 SAC
# 
# DDPG --> Deep Deterministic Policy Gradient
# Berriz Actor & Critic
# Berriz ReplayBuffer
# Berriz Epsilon Greedy Fn


def epsilon_greedy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.uniform(-2, 2)
    else:
        Q_values = epsilon(state[np.newaxis], training = False).numpy()
        return np.argmax(Q_values[0])
    
    