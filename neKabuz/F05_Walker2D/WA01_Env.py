import gymnasium as gym
import numpy as np
import pandas as pd

# https://gymnasium.farama.org/environments/mujoco/reacher/

env = gym.make("Walker2d-v5", render_mode = "human")
print(type(env.observation_space.shape[0]))
speeds = []
for i in range(5):
    obs, info = env.reset()

    for j in range(200):
        act = np.random.uniform(-1, 1, size = 6)
        obs, reward, terminated, truncated, info = env.step(act)

        speeds.append(info.get("x_velocity", None))

env.close()

# Este va a estar bien hardcore XD
# La idea es implementar SAC: Soft Actor-Critic




speeds = pd.DataFrame({"speeds": speeds})

print(speeds.describe())
