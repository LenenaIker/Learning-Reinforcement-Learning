import gymnasium as gym
import torch
import numpy as np

import os
import re
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
    MODEL_NAME = "best_ep3680_ret-2.5.pt"

    config = Config()
    start = datetime.now()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("Config:", config, "\n", "Device: ", device)


    env = gym.make(config.env_id, max_episode_steps = config.max_steps_per_episode + 1)
    eval_env = gym.make(config.env_id, max_episode_steps = config.max_steps_per_episode + 1)
    agent = TD3_Agent(env.observation_space, env.action_space, config, device)


    total_steps = 0
    best_eval = -1e9

    if MODEL_NAME is not None:
        agent.load(path = config.ckpt_dir + "/" + MODEL_NAME)
        # config.warmup_steps = int(config.total_episodes * config.max_steps_per_episode * 0.05)
        best_eval = float(re.search(r"ret(-?\d+(?:\.\d+)?)", MODEL_NAME).group(1))

        print("Model loaded: ", MODEL_NAME, " | Best Evaluation = ", best_eval)


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

            agent.push(obs, act, reward, next_obs, float(terminated)) # Usar terminated en vez de done bootstrappea en cortes por límite de tiempo
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
            print(f"Episodio {ep:03d} | Retorno: {ep_ret:8.2f}")

        if ep % config.eval_every == 0:
            avg_ret = evaluate(agent, eval_env, config.eval_episodes)
            print(f"[Eval] Episodios {ep - config.eval_every + 1}-{ep}: Retorno medio = {avg_ret:.2f}")
            if avg_ret > best_eval:
                best_eval = avg_ret
                path = os.path.join(config.ckpt_dir, f"best_ep{ep}_ret{avg_ret:.1f}.pt")
                agent.save(path)
                print(f"Guardado mejor checkpoint en {path}")
    
    
    env.close()
    eval_env.close()

    print("Tiempo total: ", datetime.now() - start)