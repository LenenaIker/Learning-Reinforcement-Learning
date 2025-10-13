from __future__ import annotations
import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
from Agent import DDPGAgent, Config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(agent: DDPGAgent, env: gym.Env, episodes: int = 5, render: bool = False) -> float:
    """Evalúa el agente sin ruido y devuelve la media de retornos."""
    returns = []
    for _ in range(episodes):
        obs, info = env.reset()
        agent.reset_noise()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done:
            act = agent.act(obs, explore = False)
            next_obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            ep_ret += float(reward)
            obs = next_obs
            steps += 1
            if render:
                env.render()
        returns.append(ep_ret)
    return float(np.mean(returns))


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make(cfg.env_id)
    eval_env = gym.make(cfg.env_id)

    assert isinstance(env.action_space, gym.spaces.Box)
    assert isinstance(env.observation_space, gym.spaces.Box)

    agent = DDPGAgent(env.observation_space, env.action_space, cfg, device)

    total_steps = 0
    best_eval = -1e9

    for ep in range(1, cfg.total_episodes + 1):
        obs, info = env.reset(seed = cfg.seed + ep)  # si quieres variabilidad por episodio
        agent.reset_noise()
        ep_ret = 0.0
        ep_len = 0

        for t in range(cfg.max_steps_per_episode):
            if total_steps < cfg.warmup_steps:
                # Acciones aleatorias durante warmup (respetando límites)
                act = env.action_space.sample().astype(np.float32)
            else:
                act = agent.act(obs, explore = True)

            next_obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            agent.push(obs, act, reward, next_obs, float(done))
            obs = next_obs
            ep_ret += float(reward)
            ep_len += 1
            total_steps += 1

            if total_steps >= cfg.warmup_steps:
                agent.train_step()

            if done:
                break

        # Decaimiento de ruido por episodio
        agent.decay_noise()

        # Logs breves en consola
        if ep % 1 == 0:
            print(f"Episodio {ep:03d} | Retorno: {ep_ret:8.2f} | pasos: {ep_len:3d} | sigma: {agent.noise.sigma if agent.noise else agent.gauss_sigma:.3f}")

        # Evaluación periódica
        if ep % cfg.eval_every == 0:
            avg_ret = evaluate(agent, eval_env, cfg.eval_episodes)
            print(f"[Eval] Episodios {ep - cfg.eval_every + 1}-{ep}: Retorno medio = {avg_ret:.2f}")
            if avg_ret > best_eval:
                best_eval = avg_ret
                path = os.path.join(cfg.ckpt_dir, f"best_ep{ep}_ret{avg_ret:.1f}.pt")
                agent.save(path)
                print(f"Guardado mejor checkpoint en {path}")

        if ep % cfg.save_every == 0:
            path = os.path.join(cfg.ckpt_dir, f"latest_ep{ep}.pt")
            agent.save(path)

    # Guardado final
    agent.save(os.path.join(cfg.ckpt_dir, "final.pt"))
    env.close(); eval_env.close()


if __name__ == "__main__":
    cfg = Config()
    print("Config:", cfg)
    start = time.time()
    train(cfg)
    print(f"Tiempo total: {time.time() - start:.1f}s")
