import gymnasium as gym
import torch

from os import listdir

from Agent import SAC
from Config import Config

config = Config()

model_names = listdir(config.ckpt_dir)

index = int(input("\n" + "\n".join([f"{i}. {s}" for i, s in enumerate(model_names, start = 1)]) + "\n\nSelect model (int):")) - 1


env = gym.make(
    id = config.env_id,
    render_mode = "human",
    max_episode_steps = config.max_steps_per_episode
)

agent = SAC(env.observation_space, env.action_space, config, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
agent.load(path = config.ckpt_dir + "/" + model_names[index])


for ep in range(10):
    obs, info = env.reset()

    for t in range(config.max_steps_per_episode):
        act = agent.act(obs = obs, explore = True)
        next_obs, reward, terminated, truncated, info = env.step(act)

        obs = next_obs