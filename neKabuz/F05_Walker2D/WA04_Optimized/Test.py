import gymnasium as gym
import mujoco as mj

import torch
import numpy as np

from os import listdir

from InputController import run_signal_builder, get_random_speed_function
from Agent import SAC
from Config import Config
from EnvWrapper import WalkerWithCommand

def set_follow_cam(env, body_name = "torso", distance = 4.5, elevation = -15, azimuth = 120):
    """
    Configura la cámara del viewer para que siga a un cuerpo del modelo MuJoCo.
    Funciona con Gymnasium + MuJoCo oficial.
    """
    viewer = None
    for attr in ["mujoco_renderer", "renderer"]:
        obj = getattr(env, attr, None) or getattr(getattr(env, "unwrapped", env), attr, None)
        if obj is not None and getattr(obj, "viewer", None) is not None:
            viewer = obj.viewer
            break
    if viewer is None:
        return False


    cam = viewer.cam
    cam.type = mj.mjtCamera.mjCAMERA_TRACKING

    model = getattr(env.unwrapped, "model", None)
    if model is None:
        return False
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return False

    cam.trackbodyid = body_id
    cam.distance = float(distance)
    cam.elevation = float(elevation)
    cam.azimuth = float(azimuth)
    return True


config = Config()

model_names = listdir(config.ckpt_dir)

index = int(input("\n" + "\n".join([f"{i}. {s}" for i, s in enumerate(model_names, start = 1)]) + "\n\nSelect model (int):")) - 1

df = run_signal_builder()

times = df["t"].to_numpy()
speeds = df["y"].to_numpy()

env = gym.make(
    id = config.env_id,
    render_mode = "human",
    max_episode_steps = config.max_steps_per_episode
)

env = WalkerWithCommand(
    env = env,
    penalty = 1.0
)


obs, info = env.reset()
_ = set_follow_cam(env, body_name = "torso", distance = 4.5, elevation = -15, azimuth = 120)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = SAC(env.observation_space, env.action_space, config, device)
agent.load(path = config.ckpt_dir + "/" + model_names[index], map_location = device)

try:
    for ep in range(3):
        obs, info = env.reset(
            seed = config.seed + ep,
            options = {
                "speed_t": times,
                "speed_y": speeds
            }
        )
        for t in range(config.max_steps_per_episode):
            act = agent.act(obs = obs, explore = True)
            next_obs, reward, terminated, truncated, info = env.step(act)

            obs = next_obs
except Exception as e:
    print(e)
finally:
    env.close()