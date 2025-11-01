
from gymnasium.spaces import Box

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import os
from dataclasses import asdict

from Config import Config
from Models import StochasticActor, Critic
from Memory import ReplayBuffer


class SAC():
    def __init__(self, obs_space: Box, act_space: Box, config: Config, device: torch.device, *args, **kwargs):
        # super(TD3_Agent, self).__init__(*args, **kwargs)
        self.config = config
        self.device = device

        # Obserbazioei abiadura gehitukoiet (input bat izangoalako), horregatik +1
        # TODO: Eztu ematen hemen danik +1 jartzeko leku honena, iwal Main-en in beharko zan
        self.obs_dim = obs_space.shape[0] + 1
        self.act_dim = act_space.shape[0]
        self.act_low = torch.as_tensor(act_space.low, dtype = torch.float32, device = device)
        self.act_high = torch.as_tensor(act_space.high, dtype = torch.float32, device = device)
        self.act_range = self.act_high - self.act_low

        self.actor = StochasticActor(self.obs_dim, self.act_dim).to(device)
        self.critic_1 = Critic(self.obs_dim, self.act_dim).to(device)
        self.critic_2 = Critic(self.obs_dim, self.act_dim).to(device)
        
        self.target_critic_1 = Critic(self.obs_dim, self.act_dim).to(device)
        self.target_critic_2 = Critic(self.obs_dim, self.act_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr = config.actor_lr)
        self.critic_1_opt = optim.Adam(self.critic_1.parameters(), lr = config.critic_lr)
        self.critic_2_opt = optim.Adam(self.critic_2.parameters(), lr = config.critic_lr)

        # SAC usa alpha cómo temperatura para controlar el equilibrio entre maximizar el reforzamiento esperado y maximizar la entropía del comportamiento.
        self.log_alpha = torch.tensor(np.log(getattr(config, "alpha_init", 0.2)), device = self.device, requires_grad = True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr = getattr(config, "alpha_lr", 3e-4))
        self.target_entropy = (-float(self.act_dim) if getattr(config, "target_entropy", None) is None else config.target_entropy)

        self.memory = ReplayBuffer(self.obs_dim, self.act_dim, size = config.buffer_size)

        self.update_step = 0


    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        self.actor.eval()
        
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype = torch.float32, device = self.device).unsqueeze(0)
            # Si explore = True: muestrea (estocástico), si no, modo determinista
            action_unit, _ = self.actor(obs_t, deterministic = not explore, with_logprob = False)  # [-1,1]
            # Reescala a [low, high]
            rescaled = (action_unit + 1) * 0.5 * self.act_range + self.act_low
            act = rescaled.squeeze(0).cpu().numpy().astype(np.float32)

        self.actor.train()
        return act


    @torch.no_grad()
    def _target(self, next_obs, rewards, dones):
        # Samplea de la política actual (no hay target actor en SAC)
        a_next_unit, log_pi_next = self.actor(next_obs, deterministic = False, with_logprob = True) # [-1,1]
        a_next = (a_next_unit + 1) * 0.5 * self.act_range + self.act_low # a escala del env

        # Q-target: min de críticos target
        q_next = torch.min(
            self.target_critic_1(next_obs, a_next),
            self.target_critic_2(next_obs, a_next)
        )
        alpha = self.log_alpha.exp()
        y = rewards + self.config.gamma * (1 - dones) * (q_next - alpha * log_pi_next)
        return y
    

    def train_step(self):
        if self.memory.len < self.config.batch_size:
            return {}

        batch = self.memory.sample(self.config.batch_size)
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)

        with torch.no_grad():
            target_q = self._target(next_obs, rews, dones)

        q1 = self.critic_1(obs, acts)
        q2 = self.critic_2(obs, acts)

        critic_loss_1 = nn.functional.mse_loss(q1, target_q)
        critic_loss_2 = nn.functional.mse_loss(q2, target_q)

        self.critic_1_opt.zero_grad()
        critic_loss_1.backward()
        nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm = 1.0)
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad()
        critic_loss_2.backward()
        nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm = 1.0)
        self.critic_2_opt.step()

        metrics = {
            "critic_1_loss": float(critic_loss_1.item()),
            "critic_2_loss": float(critic_loss_2.item())
        }

        a_unit, log_pi = self.actor(obs, deterministic = False, with_logprob = True) # [-1, 1]
        a = (a_unit + 1) * 0.5 * self.act_range + self.act_low

        q1_pi = self.critic_1(obs, a)
        q2_pi = self.critic_2(obs, a)
        q_pi = torch.min(q1_pi, q2_pi)

        alpha = self.log_alpha.exp()
        actor_loss = (alpha * log_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm = 1.0)
        self.actor_opt.step()

        metrics["actor_loss"] = float(actor_loss.item())

        # Alpha (temperatura)
        alpha_loss = ( - self.log_alpha * (log_pi + self.target_entropy).detach() ).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        metrics["alpha"] = float(alpha.item())
        metrics["alpha_loss"] = float(alpha_loss.item())


        with torch.no_grad():
            for tp, p in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                tp.data.mul_(1 - self.config.tau)
                tp.data.add_(self.config.tau * p.data)
            for tp, p in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                tp.data.mul_(1 - self.config.tau)
                tp.data.add_(self.config.tau * p.data)

        self.update_step += 1
        return metrics

    def push(self, *args, **kwargs):
        self.memory.add(*args, **kwargs)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok = True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_1_opt": self.critic_1_opt.state_dict(),
            "critic_2_opt": self.critic_2_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().item(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "cfg": asdict(self.config),
        }, path)

    def load(self, path: str, map_location = None):
        ckpt = torch.load(path, map_location = map_location)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.target_critic_1.load_state_dict(ckpt["target_critic_1"])
        self.target_critic_2.load_state_dict(ckpt["target_critic_2"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_1_opt.load_state_dict(ckpt["critic_1_opt"])
        self.critic_2_opt.load_state_dict(ckpt["critic_2_opt"])
        # alpha
        self.log_alpha.data = torch.tensor(np.log(ckpt.get("log_alpha", getattr(self.config, "alpha_init", 0.2))), device = self.device)
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
