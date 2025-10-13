import gymnasium as gym
import numpy as np

# https://gymnasium.farama.org/environments/mujoco/reacher/

env = gym.make("Reacher-v5", render_mode = "human")

for i in range(5):
    obs, info = env.reset()

    for j in range(200):
        act = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        xp = env.step(act)

        print(xp)

env.close()

