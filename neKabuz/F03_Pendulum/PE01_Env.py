import gymnasium as gym
from time import sleep
import numpy as np


MAX_EPISODE_STEPS = 15

env = gym.make("Pendulum-v1", render_mode = "human", max_episode_steps = MAX_EPISODE_STEPS + 1)

# print("Action space: ", env.action_space, " Action names:\n", env.unwrapped.get_wrapper_attr())
print("Observations/State: ", env.observation_space.shape)


obs, info = env.reset()

print("\nObs:", obs, "Info:",  info, "\n", sep = "\n")


OBS_LOWS = env.observation_space.low.astype(np.float32)
OBS_HIGHS = env.observation_space.high.astype(np.float32)

print(OBS_LOWS, OBS_HIGHS)

for i in range(MAX_EPISODE_STEPS):
    action = [np.random.uniform(-2, 2)]
    print("Action\n", action, "\n")

    obs, reward, terminated, truncated, info = env.step(np.array(action, dtype = np.float32))

    print(f"Step {i}\nReward {reward}\nInfo: {info}\n\n")

    if terminated or truncated:
        print(f"Terminado {terminated}\nTruncado {truncated}")
        break

    sleep(1.5)


env.close()



# Action: Float bat pasa behar diot. -2 ta 2 bitartean, positiboa bada Antihorario, bestela Horario
# Reward: Zutik badao penduloa 0, geroz ta hurrutio eon zutik eotetik, geroz ta zigor haundiagoa.

# Interesgarria da torkea dakan lehen ingurumena izangoalako.