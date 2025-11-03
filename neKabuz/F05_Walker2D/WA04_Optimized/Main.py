import gymnasium as gym

import random
import numpy as np
import torch
torch.set_float32_matmul_precision("medium")

from datetime import datetime
import re
import os

from Agent import SAC
from Config import Config
from InputController import get_random_speed_function
from EnvWrapper import WalkerWithCommand


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(agent, eval_env, episodes=5):
    total_reward = 0
    steps = 0
    obs, _ = eval_env.reset()
    done = np.zeros(eval_env.num_envs, dtype=bool)

    while not np.all(done):
        act = agent.act(obs, explore=False)
        obs, reward, terminated, truncated, _ = eval_env.step(act)
        done |= (terminated | truncated)
        total_reward += reward.sum()
        steps += 1

    avg_reward = total_reward / episodes
    return avg_reward


def make_env(seed, n_speeds):
    def thunk():
        e = gym.make(config.env_id, max_episode_steps = config.max_steps_per_episode + 1)
        e = WalkerWithCommand(
            env = e,
            speed_function = get_random_speed_function(n_speeds),
            n_speeds = n_speeds,
            penalty = 1.0
        )
        e.reset(seed = seed)
        return e
    return thunk



if __name__ == "__main__":
    MODEL_NAME = None
    N_SPEEDS = 10

    config = Config()
    start = datetime.now()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Config:", config, "\n", "Device:", device, sep = "\n")

    num_train_envs = 8
    num_eval_envs = 4

    env = gym.vector.AsyncVectorEnv(
        [make_env(config.seed + i, N_SPEEDS, eval_mode=False) for i in range(num_train_envs)]
    )

    eval_env = gym.vector.SyncVectorEnv(  # Sync para eval → resultados más reproducibles
        [make_env(config.seed + 10 + i, N_SPEEDS, eval_mode=True) for i in range(num_eval_envs)]
    )


    agent = SAC(env.observation_space, env.action_space, config, device)

    total_steps = 0
    best_eval = -1e9
    speed = 0

    if MODEL_NAME is not None:
        agent.load(path = config.ckpt_dir + "/" + MODEL_NAME)
        # config.warmup_steps = int(config.total_episodes * config.max_steps_per_episode * 0.05)
        best_eval = float(re.search(r"ret(-?\d+(?:\.\d+)?)", MODEL_NAME).group(1))

        print("Model loaded: ", MODEL_NAME, " | Best Evaluation = ", best_eval)


    for ep in range(1, config.total_episodes + 1):
        obs, info = env.reset(
            speed_function = get_random_speed_function(N_SPEEDS),
            n_speeds = N_SPEEDS,
            seed = config.seed + ep
        )

        ep_ret = 0.0
        ep_len = 0
        for t in range(config.max_steps_per_episode):
            if total_steps < config.warmup_steps:
                act = env.action_space.sample().astype(np.float32)
            else:
                act = agent.act(obs, explore = True)
            

            next_obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            agent.push(obs, act, reward, next_obs, float(terminated))
            obs = next_obs
            ep_ret += float(reward)
            ep_len += 1
            total_steps += 1

            if total_steps >= config.warmup_steps:
                for _ in range(config.updates_per_step):
                    agent.train_step()

            if done:
                break

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