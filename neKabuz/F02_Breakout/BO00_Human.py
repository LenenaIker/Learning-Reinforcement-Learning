import time
import numpy as np
import keras
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation
from gymnasium.utils.save_video import save_video

MODEL_PATH = "neKabuz/F02_Breakout/BO08_75e.keras"
SCREEN_SIZE = 84
FRAME_SKIP = 4
FRAME_STACK = 4
SEED = 0
MAX_STEPS = 5000
SLEEP = 1/60


model = keras.models.load_model(MODEL_PATH)

env = gym.make(
    "ALE/Breakout-v5",
    render_mode = "rgb_array_list",
    frameskip = 1,
    max_episode_steps = MAX_STEPS + 1
)
env = AtariPreprocessing(
    env,
    screen_size = SCREEN_SIZE,
    grayscale_obs = True,
    frame_skip = FRAME_SKIP,
    noop_max = 30
)
env = FrameStackObservation(env, stack_size = FRAME_STACK)

env = TransformObservation(
    env = env,
    func = lambda obs: np.transpose(obs, (1, 2, 0)),
    observation_space = gym.spaces.Box(
        low = 0,
        high = 255,
        shape = (SCREEN_SIZE, SCREEN_SIZE, FRAME_STACK),
        dtype = np.uint8
    )
)

obs, _ = env.reset(seed = SEED)

def preprocess(x): return x

def logits_to_action(pi):
    if isinstance(pi, (list, tuple)) and len(pi) > 1:
        pi = pi[0]

    if pi.ndim == 1:
        return int(np.argmax(pi))
    else:
        return int(np.argmax(pi[0]))

for t in range(MAX_STEPS):
    x = preprocess(obs)[np.newaxis, ...]
    pred = model(x)
    
    if hasattr(pred, "numpy"):
        pred = pred.numpy()
    elif isinstance(pred, (list, tuple)):
        pred = [p.numpy() if hasattr(p, "numpy") else p for p in pred]

    action = logits_to_action(pred)

    obs, reward, terminated, truncated, info = env.step(action)

    time.sleep(SLEEP)

    if terminated or truncated:
        save_video(
            frames = env.render(),
            video_folder = "videos",
            fps = env.metadata["render_fps"]
        )

        obs, _ = env.reset()

    print(t)
        

env.close()
