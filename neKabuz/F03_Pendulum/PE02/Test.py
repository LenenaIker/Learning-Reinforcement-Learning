import gymnasium as gym
import torch

from Agent import DDPGAgent
from Config import Config


PATH_MODEL = r"neKabuz\F03_Pendulum\PE02\checkpoints_ddpg_pendulum\final.pt"

cfg = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make(
    id = cfg.env_id,
    render_mode = "human"
)


agent = DDPGAgent(env.observation_space, env.action_space, cfg, device)
agent.load(path = PATH_MODEL)


for ep in range(10):
    obs, info = env.reset()

    for t in range(cfg.max_steps_per_episode):
        act = agent.act(obs = obs, explore = True)
        next_obs, reward, terminated, truncated, info = env.step(act)

        obs = next_obs