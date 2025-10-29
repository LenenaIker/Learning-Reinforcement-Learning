import gymnasium as gym
import numpy as np
 
# https://gymnasium.farama.org/environments/mujoco/reacher/

env = gym.make("Walker2d-v5", render_mode = "human")

for i in range(5):
    obs, info = env.reset()

    for j in range(200):
        act = np.random.uniform(-1, 1, size = 6)
        xp = env.step(act)


        print(xp)

env.close()

# Este va a estar bien hardcore XD
# La idea es implementar SAC: Soft Actor-Critic

