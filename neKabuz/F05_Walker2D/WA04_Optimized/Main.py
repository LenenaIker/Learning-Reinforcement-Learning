import gymnasium as gym
import numpy as np
import torch
import random
from datetime import datetime
import os
import re

from Agent import SAC
from Config import Config
from InputController import random_smooth_speed_arrays
from EnvWrapper import WalkerWithCommand

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env(seed, n_speeds, config):
    def thunk():
        env = gym.make(
            config.env_id,
            max_episode_steps = config.max_steps_per_episode,
        )
        env = WalkerWithCommand(
            env = env,
            n_speeds = n_speeds,
            penalty = 1.0
        )
        env.reset(seed = seed)
        return env
    return thunk

def evaluate(agent, eval_env, episodes = 5, n_speeds = 10):
    n_envs = eval_env.num_envs
    returns = []
    for _ in range(episodes):
        times = np.linspace(0.0, 10.0, n_speeds, dtype=np.float32)
        speeds = np.random.uniform(-1.0, 1.0, size=n_speeds).astype(np.float32)
        obs, _ = eval_env.reset(
            options = {
                "speed_t": times,
                "speed_y": speeds,
                "n_speeds": n_speeds
            }
        )
        
        done = np.zeros(n_envs, dtype = bool)
        ep_ret = np.zeros(n_envs, dtype = np.float32)

        while not np.all(done):
            act = agent.act(obs, explore = False)
            obs, reward, terminated, truncated, _ = eval_env.step(act)
            dones = np.logical_or(terminated, truncated)

            ep_ret += reward * (~done)
            done |= dones

        returns.append(ep_ret.mean())
    return float(np.mean(returns))

if __name__ == "__main__":
    MODEL_NAME = None
    N_SPEEDS = 10

    config = Config()
    start = datetime.now()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Config:", config, "\n", "Device:", device, sep="\n")

    num_train_envs = 16
    num_eval_envs = 4

    train_env = gym.vector.AsyncVectorEnv([make_env(config.seed + i, N_SPEEDS, config) for i in range(num_train_envs)])
    eval_env = gym.vector.SyncVectorEnv([make_env(config.seed + 100 + i, N_SPEEDS, config) for i in range(num_eval_envs)])

    agent = SAC(train_env.single_observation_space, train_env.single_action_space, config, device)

    total_steps = 0
    best_eval = -1e9

    if MODEL_NAME is not None:
        agent.load(os.path.join(config.ckpt_dir, MODEL_NAME))
        best_eval = float(re.search(r"ret(-?\d+(?:\.\d+)?)", MODEL_NAME).group(1))
        print("Model loaded:", MODEL_NAME, "| Best Evaluation =", best_eval)

    for ep in range(1, config.total_episodes + 1):
        times, speeds = random_smooth_speed_arrays(N_SPEEDS)
        # No permite enviar funciones en options, por lo que mando los datos y que genere la función en el otro lado
        obs, info = train_env.reset(
            seed = config.seed + ep,
            options = {
                "speed_t": times,
                "speed_y": speeds,
                "n_speeds": N_SPEEDS
            }
        )

        done = np.zeros(num_train_envs, dtype = bool)
        ep_ret = np.zeros(num_train_envs, dtype = np.float32)

        for t in range(config.max_steps_per_episode):
            if total_steps < config.warmup_steps:
                act = np.stack(
                    [train_env.single_action_space.sample()
                     for _ in range(num_train_envs)]
                ).astype(np.float32)
            else:
                act = agent.act(obs, explore = True)

            next_obs, reward, terminated, truncated, info = train_env.step(act)
            dones = np.logical_or(terminated, truncated)

            agent.push(obs, act, reward, next_obs, dones)

            obs = next_obs
            ep_ret += reward
            done |= dones

            total_steps += num_train_envs

            if total_steps >= config.warmup_steps:
                for _ in range(config.updates_per_step):
                    agent.train_step()

            if np.all(done):
                break

        print(f"Episodio {ep:03d} | Retorno medio: {ep_ret.mean():8.2f}")

        if ep % config.eval_every == 0:
            avg_ret = evaluate(agent, eval_env, config.eval_episodes, N_SPEEDS)
            print(f"[Eval] Episodios {ep - config.eval_every + 1}-{ep}: Retorno medio = {avg_ret:.2f}")
            if avg_ret > best_eval:
                best_eval = avg_ret
                os.makedirs(config.ckpt_dir, exist_ok = True)
                path = os.path.join(config.ckpt_dir, f"best_ep{ep}_ret{avg_ret:.1f}.pt")
                agent.save(path)
                print(f"Guardado mejor checkpoint en {path}")

    train_env.close()
    eval_env.close()
    print("Tiempo total:", datetime.now() - start)
