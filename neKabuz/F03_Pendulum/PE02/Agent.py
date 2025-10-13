from __future__ import annotations
import os
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Models import Actor, Critic
from Config import Config
from ReplayBuffer import ReplayBuffer
from Noise import OUNoise


class DDPGAgent:
    def __init__(self, obs_space: gym.spaces.Box, act_space: gym.spaces.Box, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.shape[0]
        self.act_low = torch.as_tensor(act_space.low, dtype = torch.float32, device = device)
        self.act_high = torch.as_tensor(act_space.high, dtype = torch.float32, device = device)

        self.actor = Actor(self.obs_dim, self.act_dim).to(device)
        self.critic = Critic(self.obs_dim, self.act_dim).to(device)
        self.target_actor = Actor(self.obs_dim, self.act_dim).to(device)
        self.target_critic = Critic(self.obs_dim, self.act_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr = cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr = cfg.critic_lr)

        self.replay = ReplayBuffer(self.obs_dim, self.act_dim, size = cfg.buffer_size)

        if cfg.use_ou_noise:
            self.noise = OUNoise(self.act_dim, sigma = cfg.noise_sigma)
        else:
            self.noise = None
        self.gauss_sigma = cfg.noise_sigma

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        self.actor.eval() # Poner el actor modo evaluación
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype = torch.float32, device = self.device).unsqueeze(0)
            # Actor produce acción no acotada -> tanh -> reescala a [low, high]
            raw = self.actor(obs_t)
            a = torch.tanh(raw)
            # reescala
            act = (a + 1) * 0.5 * (self.act_high - self.act_low) + self.act_low
            act = act.squeeze(0).cpu().numpy()
        self.actor.train() # Poner el actor modo entrenamiento

        if explore:
            if self.noise:
                act = act + self.noise.sample()
            else:
                act = act + np.random.normal(0, self.gauss_sigma, size = act.shape)
        # clip a límites del entorno
        act = np.clip(act, self.act_low.cpu().numpy(), self.act_high.cpu().numpy())
        return act.astype(np.float32)

    @torch.no_grad()
    def _target(self, next_obs, rewards, dones):
        # y = r + gamma * (1-done) * Q'(s', μ'(s'))
        raw_next = self.target_actor(next_obs)
        a_next = torch.tanh(raw_next)
        a_next = (a_next + 1) * 0.5 * (self.act_high - self.act_low) + self.act_low
        q_next = self.target_critic(next_obs, a_next)
        y = rewards + self.cfg.gamma * (1 - dones) * q_next
        return y

    def train_step(self):
        if self.replay.len < self.cfg.batch_size:
            return {}
        batch = self.replay.sample(self.cfg.batch_size)
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)

        # Critic loss
        with torch.no_grad():
            target_q = self._target(next_obs, rews, dones)
        q = self.critic(obs, acts)
        critic_loss = nn.functional.mse_loss(q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm = 1.0)
        self.critic_opt.step()

        # Actor loss: maximizar Q(s, μ(s)) -> minimizar -Q
        raw = self.actor(obs)
        a = torch.tanh(raw)
        a = (a + 1) * 0.5 * (self.act_high - self.act_low) + self.act_low
        actor_loss = -self.critic(obs, a).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm = 1.0)
        self.actor_opt.step()

        # Soft update
        with torch.no_grad():
            for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
                tp.data.mul_(1 - self.cfg.tau)
                tp.data.add_(self.cfg.tau * p.data)
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - self.cfg.tau)
                tp.data.add_(self.cfg.tau * p.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def push(self, *args, **kwargs):
        self.replay.add(*args, **kwargs)

    def reset_noise(self):
        if self.noise:
            self.noise.reset()

    def decay_noise(self):
        if self.noise:
            self.noise.sigma = max(self.cfg.noise_min_sigma, self.noise.sigma * self.cfg.noise_decay)
        else:
            self.gauss_sigma = max(self.cfg.noise_min_sigma, self.gauss_sigma * self.cfg.noise_decay)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok = True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'cfg': asdict(self.cfg),
        }, path)

    def load(self, path: str, map_location = None):
        ckpt = torch.load(path, map_location = map_location)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_actor.load_state_dict(ckpt['target_actor'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])

