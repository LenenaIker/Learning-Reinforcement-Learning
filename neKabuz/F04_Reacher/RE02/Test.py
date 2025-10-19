import gymnasium as gym
import torch

from Agent import TD3_Agent
from Config import Config

models = [
    "best_ep710_ret-3.5.pt",
    "best_ep790_ret-2.9.pt",
    "best_ep890_ret-3.0.pt",
    "best_ep1000_ret-2.8.pt",
    "best_ep3680_ret-2.5.pt"
]


MODEL_NAME = models[4]

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make(
    id = config.env_id,
    render_mode = "human"
)

agent = TD3_Agent(env.observation_space, env.action_space, config, device)
agent.load(path = config.ckpt_dir + "/" + MODEL_NAME)


for ep in range(10):
    obs, info = env.reset()

    for t in range(config.max_steps_per_episode):
        act = agent.act(obs = obs, explore = True)
        next_obs, reward, terminated, truncated, info = env.step(act)

        obs = next_obs