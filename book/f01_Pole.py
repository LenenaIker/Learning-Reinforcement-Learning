import gymnasium as gym
import numpy as np


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

env = gym.make("CartPole-v1", render_mode = "human")


totals = []
for episode in range(500):
    episode_rewards = 0
    obs, info = env.reset()
    
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_rewards += reward
        # print(reward, "", episode, "\n")

        if terminated or truncated:
            break
    
    totals.append(episode_rewards)

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))

env.close()
