import gymnasium as gym
import ale_py

import keras

import numpy as np
from collections import deque

from datetime import datetime
import os


# Gauza simple bat iten saiatuko naiz hasieran, zuzenean Double DQNkin hasi ordez, modelo bakarra erabilikot. f06_MountainCar.pyn bezala.
# Entorno hau ezta MountainCar bezain xurra, nahiko eskuzabala danez, ezta hain zaia izango sariak lortzea
# Webgune ta foro batzutan esateute lehen saiakeretan pelota sakatzeun akzioa artifizialki jartzeko. Eztet ingo.
 

MODEL_PATH = "neKabuz\F02_Breakout\BO02_LehenSaiakera.keras" # .h5 etik .keras-era
SAVED_MODEL = os.path.exists(MODEL_PATH)


EPISODES = 500
MAX_EPISODE_STEPS = 100

TARGET_UPDATE_EVERY_STEPS = 2000

REPLAY_CAPACITY = 100_000
BATCH_SIZE = 32

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 5e-4

SUPER_VERBOSE = False


print("Modelo encontrado: ", SAVED_MODEL)


gym.register_envs(ale_py)
env = gym.make(
    "ALE/Breakout-v5",
    render_mode = "human" if SUPER_VERBOSE else None,
    max_episode_steps = MAX_EPISODE_STEPS + 1
)

OBS_LOWS = env.observation_space.low.astype(np.float32)
OBS_HIGHS = env.observation_space.high.astype(np.float32)

# Hau aldatu ingot Scikit minmaxScaler batekin edo keras layers normalization batekin
def normalize(state):
    scale = np.where((OBS_HIGHS - OBS_LOWS) == 0, 1.0, (OBS_HIGHS - OBS_LOWS))
    return ((state - OBS_LOWS) / scale * 2.0 - 1.0).astype(np.float32)

model = None
if SAVED_MODEL:
    model = keras.models.load_model(MODEL_PATH)
else:
    model = keras.models.Sequential([
        keras.Input(shape = env.observation_space.shape),
        keras.layers.Dense(32, activation = "elu"),
        keras.layers.Dense(32, activation = "elu"),
        keras.layers.Dense(env.action_space.n)
    ])

loss_fn = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate = 1e-4)
replay_buffer = deque(maxlen = REPLAY_CAPACITY)


all_rewards = []
all_losses = []

start = datetime.now()
last_time = start
for episode in range(EPISODES):
    current_state, info = env.reset()
    current_state = normalize(current_state)

    total_reward = 0
    losses = []

    for step in range(MAX_EPISODE_STEPS):

        action = np.random.randint(env.action_space.n)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = normalize(next_state)


        total_reward += reward
        replay_buffer.append((current_state, action, reward, next_state, terminated, truncated))
        current_state = next_state

        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, terminateds, truncateds = sample_experiences(batch_size)
            dones = np.logical_or(terminateds, truncateds).astype(np.float32)
            