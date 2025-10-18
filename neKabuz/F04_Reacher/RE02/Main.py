import gymnasium as gym
import torch
import numpy as np

import os
import random
from datetime import datetime

from Agent import TD3_Agent
from Config import Config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(agent: TD3_Agent, env: gym.Env, episodes: int = 5, render: bool = False) -> float:
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


if __name__ == "__main__":
    config = Config()
    start = datetime.now()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("Config:", config, "\n", "Device: ", device)


    env = gym.make(config.env_id)
    eval_env = gym.make(config.env_id)
    agent = TD3_Agent(env.observation_space, env.action_space, config, device)

    agent.load()

    total_steps = 0
    best_eval = -1e9
    for ep in range(1, config.total_episodes + 1):
        obs, info = env.reset(seed = config.seed + ep)
        agent.reset_noise()
        
        ep_ret = 0.0
        ep_len = 0
        for t in range(config.max_steps_per_episode):
            if total_steps < config.warmup_steps:
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

            if total_steps >= config.warmup_steps:
                agent.train_step()

            if done:
                break

        agent.decay_noise()

        if ep % 1 == 0:
            print(f"Episodio {ep:03d} | Retorno: {ep_ret:8.2f} | pasos: {ep_len:3d} | sigma: {agent.ou_sigma if config.use_ou_noise else agent.gauss_sigma:.3f}")

        if ep % config.eval_every == 0:
            avg_ret = evaluate(agent, eval_env, config.eval_episodes)
            print(f"[Eval] Episodios {ep - config.eval_every + 1}-{ep}: Retorno medio = {avg_ret:.2f}")
            if avg_ret > best_eval:
                best_eval = avg_ret
                path = os.path.join(config.ckpt_dir, f"best_ep{ep}_ret{avg_ret:.1f}.pt")
                agent.save(path)
                print(f"Guardado mejor checkpoint en {path}")
    
    agent.save(os.path.join(config.ckpt_dir, "final.pt"))
    
    env.close()
    eval_env.close()

    print("Tiempo total: ", datetime.now() - start)