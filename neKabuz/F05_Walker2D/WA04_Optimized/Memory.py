import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int = int(1e6)):
        self.size = size
        self.ptr = 0
        self.len = 0

        self.obs = np.zeros((size, obs_dim), dtype = np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype = np.float32)
        self.acts = np.zeros((size, act_dim), dtype = np.float32)
        self.rews = np.zeros((size, 1), dtype = np.float32)
        self.dones = np.zeros((size, 1), dtype = np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.add_batch(
            obs[np.newaxis, :],
            act[np.newaxis, :],
            np.array([[rew]], dtype = np.float32),
            next_obs[np.newaxis, :],
            np.array([[done]], dtype = np.float32),
        )

    def add_batch(self, obs, acts, rews, next_obs, dones):
        n = obs.shape[0]
        idxs = (np.arange(n) + self.ptr) % self.size

        self.obs[idxs] = obs
        self.acts[idxs] = acts
        self.rews[idxs] = rews.reshape(n, 1)
        self.next_obs[idxs] = next_obs
        self.dones[idxs] = dones.reshape(n, 1)

        self.ptr = (self.ptr + n) % self.size
        self.len = min(self.len + n, self.size)

    def sample(self, batch_size: int = 256, device: torch.device | None = None):
        idxs = np.random.randint(0, self.len, size = batch_size)

        obs = torch.as_tensor(self.obs[idxs], dtype = torch.float32)
        acts = torch.as_tensor(self.acts[idxs], dtype = torch.float32)
        rews = torch.as_tensor(self.rews[idxs], dtype = torch.float32)
        next_obs = torch.as_tensor(self.next_obs[idxs], dtype = torch.float32)
        dones = torch.as_tensor(self.dones[idxs], dtype = torch.float32)

        if device is not None:
            obs = obs.to(device, non_blocking = True)
            acts = acts.to(device, non_blocking = True)
            rews = rews.to(device, non_blocking = True)
            next_obs = next_obs.to(device, non_blocking = True)
            dones = dones.to(device, non_blocking = True)

        return {
            "obs": obs,
            "acts": acts,
            "rews": rews,
            "next_obs": next_obs,
            "dones": dones,
        }
