
from gymnasium.spaces import Box

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import os
from dataclasses import asdict

from Config import Config
from Models import Actor, Critic
from Memory import ReplayBuffer
from Noise import OUActionNoise, GaussianActionNoise



class TD3_Agent(nn.Module):
    def __init__(self, obs_space: Box, act_space: Box, config: Config, device: torch.device, *args, **kwargs):
        super(TD3_Agent, self).__init__(*args, **kwargs)
        self.config = config
        self.device = device

        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.shape[0]
        self.register_buffer("act_low",  torch.as_tensor(act_space.low,  dtype = torch.float32))
        self.register_buffer("act_high", torch.as_tensor(act_space.high, dtype = torch.float32))
        self.register_buffer("act_range", self.act_high - self.act_low)
        # Register_buffer sirve para que viajen de CPU <=> GPU y mantengan dtype

        self.actor = Actor(self.obs_dim, self.act_dim).to(device)
        self.critic_1 = Critic(self.obs_dim, self.act_dim).to(device)
        self.critic_2 = Critic(self.obs_dim, self.act_dim).to(device)
        
        self.target_actor = Actor(self.obs_dim, self.act_dim).to(device)
        self.target_critic_1 = Critic(self.obs_dim, self.act_dim).to(device)
        self.target_critic_2 = Critic(self.obs_dim, self.act_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())


        self.actor_opt = optim.Adam(self.actor.parameters(), lr = config.actor_lr)
        self.critic_1_opt = optim.Adam(self.critic_1.parameters(), lr = config.critic_lr)
        self.critic_2_opt = optim.Adam(self.critic_2.parameters(), lr = config.critic_lr)


        self.memory = ReplayBuffer(self.obs_dim, self.act_dim, size = config.buffer_size)


        if config.use_ou_noise:
            self.ou_sigma = config.noise_sigma
            self.noise = OUActionNoise(
                mean = np.zeros(self.act_dim, dtype = np.float32),
                std_deviation = np.ones(self.act_dim, dtype = np.float32) * self.ou_sigma,
                theta = 0.15,
                dt = 1e-2
            )
        else:
            self.gauss_sigma = config.noise_sigma
            self.noise = GaussianActionNoise(
                mean = np.zeros(self.act_dim, dtype = np.float32),
                std = np.ones(self.act_dim, dtype = np.float32) * self.gauss_sigma
            )

        self.update_step = 0




    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        self.actor.eval() # Modo Evaluación == Desactivar dropout, politica determinista...

        with torch.no_grad(): # Efizientzia kontuak: pytorchek be kabuz tensoreekin zerikusiakan dena gordetzeu ezbazu hau jartzen. Eztu ezertako balio, baino ezbaezu jartzen mantxogo jungoa
            obs_t = torch.as_tensor(obs, dtype = torch.float32, device = self.device).unsqueeze(0)

            unscaled_action = self.actor(obs_t)
            scaled_action = torch.tanh(unscaled_action) # Escalar a [-1, 1]
            rescaled_action = (scaled_action + 1) * 0.5 * self.act_range + self.act_low # Reescalar a [act_low, act_high]
            act = rescaled_action.squeeze(0).cpu().numpy()

        self.actor.train() # Modo Entrenamiento

        if explore and self.noise is not None:
            act = act + self.noise()

        # Limitar la acción a [act_low, act_high], por si el noise nos lo ha movido fuera del rango posible
        act = np.clip(act, self.act_low.cpu().numpy(), self.act_high.cpu().numpy())
        return act.astype(np.float32)


    @torch.no_grad()
    def _target(self, next_obs, rewards, dones):
        """
        TD3 target:
        y = r + gamma * (1 - done) * min(Q1'(s', a'_noisy), Q2'(s', a'_noisy))
        con policy smoothing (ruido gaussiano truncado) y recorte a los límites de acción.
        """

        # Target policy: Predicción de la siguiente acción. Escalado a [act_low, act_high]
        raw_next = self.target_actor(next_obs)
        act_next = torch.tanh(raw_next)
        act_next = (act_next + 1) * 0.5 * self.act_range + self.act_low

        # Policy smoothing: ruido gaussiano clipeado a [act_low, act_high]
        # act_next_noisy == a'_noisy
        noise = torch.randn_like(act_next) * (self.config.policy_noise * self.act_range)
        noise = torch.clamp(
            noise,
            -self.config.noise_clip * self.act_range,
            self.config.noise_clip * self.act_range
        )

        act_next_noisy = act_next + noise
        act_next_noisy = torch.clamp(act_next_noisy, self.act_low, self.act_high)
        
        # Twin critics target y mínimo: Esto es lo que minimiza el positive bias/overestimation.
        # q_next == min(Q1'(s', a'_noisy), Q2'(s', a'_noisy))
        q_next = torch.min(
            self.target_critic_1(next_obs, act_next_noisy),
            self.target_critic_2(next_obs, act_next_noisy)
        )

        # y = r + gamma * (1 - done) * q_next
        y = rewards + self.config.gamma * (1 - dones) * q_next
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

        # Critic loss
        with torch.no_grad():
            target_q = self._target(next_obs, rews, dones)
        # Lehen kritikoai gradientea kalkulatu
        q_1 = self.critic_1(obs, acts)
        critic_loss_1 = nn.functional.mse_loss(q_1, target_q) # akatsak neurtu
        self.critic_1_opt.zero_grad() # Limpiar los gradientes previos
        critic_loss_1.backward() # pyTorchek be kabuz tensoreekin zerikusiakan guztia gordetzeunez, zuzenean gradienteak kalkulatu ta aplikatzeko
        nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm = 1.0) # Clipear gradientes
        self.critic_1_opt.step()

        # Berdina bigarren kritikoakin
        q_2 = self.critic_2(obs, acts)
        critic_loss_2 = nn.functional.mse_loss(q_2, target_q)
        self.critic_2_opt.zero_grad()
        critic_loss_2.backward()
        nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm = 1.0)
        self.critic_2_opt.step()

        metrics = {
            "critic_1_loss": float(critic_loss_1.item()),
            "critic_2_loss": float(critic_loss_2.item())
        }

        # Actualización con DELAY del actor + soft updates. De este delay viene la D del nombre TD3.
        if self.update_step % self.config.policy_freq == 0:
            act = torch.tanh(self.actor(obs)) # act [-1, 1]
            act = (act + 1) * 0.5 * (self.act_high - self.act_low) + self.act_low # act [act_low, act_high]

            actor_loss = -self.critic_1(obs, act).mean()
            # ¿Porqué se usa solo el CRITIC_1 para actualizar el ACTOR?
            # CRITIC_2 se añade para evitar el sesgo positivo o sobreestimación del primer CRITIC a la hora de calcular el TARGET
            # Si actualizamos el actor usando MIN(value_1, value_2), estaremos añadiendo la posibilidad de un sesgo negativo.
            # Es decir, nuestra politica podría ser demasiado conservadora.
            # 
            # Fujimoto, Scott, et al. "Addressing Function Approximation Error in Actor-Critic Methods" (2018):
            # <<We use only the first critic to update the actor to avoid introducing unnecessary bias in the policy gradient.>>
            #  
            # Hay alguna variante de TD3, como "TD3-Averaged Critics", donde usan la combinación de ambos criticos para la pólitica
            # Pero parece ser que no siempre mejoran los resultados además de añadir computo y necesidad de tuning más refinado.


            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm = 1.0)
            self.actor_opt.step()

            with torch.no_grad():
                for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
                    tp.data.mul_(1 - self.config.tau)
                    tp.data.add_(self.config.tau * p.data)
                
                for tp, p in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                    tp.data.mul_(1 - self.config.tau)
                    tp.data.add_(self.config.tau * p.data)
                
                for tp, p in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                    tp.data.mul_(1 - self.config.tau)
                    tp.data.add_(self.config.tau * p.data)

            
            metrics["actor_loss"] = float(actor_loss.item())

        self.update_step += 1

        return metrics
    

    def push(self, *args, **kwargs):
        self.memory.add(*args, **kwargs)

    def reset_noise(self):
        if self.noise is not None:
            self.noise.reset()

    def decay_noise(self):
        if isinstance(self.noise, OUActionNoise):
            self.ou_sigma = max(self.config.noise_min_sigma, self.ou_sigma * self.config.noise_decay)
            self.noise.std_deviation = np.ones_like(self.noise.std_deviation) * self.ou_sigma
        elif isinstance(self.noise, GaussianActionNoise):
            self.gauss_sigma = max(self.config.noise_min_sigma, self.gauss_sigma * self.config.noise_decay)
            self.noise.std = np.ones_like(self.noise.std) * self.gauss_sigma



    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok = True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_1_opt": self.critic_1_opt.state_dict(),
            "critic_2_opt": self.critic_2_opt.state_dict(),
            "cfg": asdict(self.config),
        }, path)

    def load(self, path: str, map_location = None):
        ckpt = torch.load(path, map_location = map_location)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.target_critic_1.load_state_dict(ckpt["target_critic_1"])
        self.target_critic_2.load_state_dict(ckpt["target_critic_2"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_1_opt.load_state_dict(ckpt["critic_1_opt"])
        self.critic_2_opt.load_state_dict(ckpt["critic_2_opt"])

