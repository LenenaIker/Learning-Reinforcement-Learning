import gymnasium as gym
import ale_py

import keras
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import numpy as np
from collections import deque

from datetime import datetime
import os

import matplotlib.pyplot as plt

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
INFO_EACH_EPISODES = 2 if SUPER_VERBOSE else 15

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

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size = batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, terminateds, truncateds = [
        np.array([ experience[field_index] for experience in batch ]) for field_index in range(6)
    ]
    return states, actions, rewards, next_states, terminateds, truncateds



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
            states, actions, rewards, next_states, terminateds, truncateds = sample_experiences(BATCH_SIZE)
            dones = np.logical_or(terminateds, truncateds).astype(np.float32)
            next_Q_values = model.predict(next_states, verbose = 0)

            max_next_Q_values = np.max(next_Q_values, axis = 1)
            target_Q_values = (rewards + (1 - dones) * DISCOUNT_FACTOR * max_next_Q_values)
            target_Q_values = target_Q_values[:, np.newaxis]

            mask = tf.one_hot(actions, env.action_space.n)

            with tf.GradientTape() as tape:
                all_Q_values = model(states, training = True)
                Q_values = tf.reduce_sum(all_Q_values * mask, axis = 1, keepdims = True)
                loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            losses.append(loss.numpy())

        if terminated or truncated:
            print(f"Terminado {terminated}   Truncado {truncated}")
            break

    all_rewards.append(total_reward)
    if losses:
        all_losses.append(np.mean(losses))

    if episode % INFO_EACH_EPISODES == 0:
        mean_reward = np.mean(all_rewards[-10:])
        mean_loss = np.mean(all_losses[-10:]) if all_losses else 0
        
        print(f"\nEpisodio {episode:3d}: Recompensa media = {mean_reward:.1f}\nε = {epsilon:.2f}, pérdida media ≈ {mean_loss:.4f}\nTiempo: {datetime.now() - last_time if last_time else None}")
        last_time = datetime.now()
        

env.close()

print("Fin del programa, Tiempo: ", datetime.now() - start)

model.save(MODEL_PATH)

plt.plot(all_rewards)
plt.xlabel("Episodios")
plt.ylabel("Recompensa total")
plt.title("Evolución de la recompensa")
plt.show()
