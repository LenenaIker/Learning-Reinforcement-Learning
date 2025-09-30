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
MAX_EPISODE_STEPS = 5000

TARGET_UPDATE_EVERY_STEPS = 2000

REPLAY_CAPACITY = 100_000
BATCH_SIZE = 32

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 1e-4

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


# Emateunez hau RLn asko erabiltzea, alternatiba gutxi ikusiitut oaingoz.
def epsilon_greedy_policy(state, epsilon = 0):
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    else:
        Q_values = model(state[np.newaxis], training = False).numpy()
        return np.argmax(Q_values[0])

def sample_experiences(batch_size):
    replay_array = np.array(replay_buffer, dtype = object)
    indices = np.random.randint(len(replay_buffer), size = batch_size)
    batch = [replay_array[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([ experience[field_index] for experience in batch ]) for field_index in range(5)
    ]
    return states, actions, rewards, next_states, dones


#   Conv2D() parametros interesantes:
#   filters: número de filtros que aprende la capa
#       cada filtro es un pequeño kernel que detecta un tipo de patrón (borde verical, movimiento...)
#   kernel_size: tamaño de filtro, cuantos píxeles de entrada abarca cada "receptivo". Grandes para patrones globales y pequeños para detalles
#   strides: Cuanto más grande sea más reduce la imágen inicial.
#   padding: Cómo tratar bordes. En atari se usa "valid"
#       "valid" = sin padding, la salida se reduce.
#       "same" = añade ceros al rededor para mantener dimensiones

model = None
if SAVED_MODEL:
    model = keras.models.load_model(MODEL_PATH)
else:
    model = keras.models.Sequential([
        keras.Input(shape = env.observation_space.shape),
        keras.layers.Lambda(lambda t: tf.cast(t, tf.float32) / 255.0),
        keras.layers.Conv2D(32, 8, strides = 4, activation = "relu"),
        keras.layers.Conv2D(64, 4, strides = 2, activation = "relu"),
        keras.layers.Conv2D(64, 3, strides = 1, activation = "relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation = "elu"),
        keras.layers.Dense(256, activation = "elu"),
        keras.layers.Dense(env.action_space.n, activation = 'softmax')
    ])

# Double DQN
target_model = keras.models.clone_model(model)
_ = model(np.zeros((1,) + env.observation_space.shape, dtype = np.float32))
_ = target_model(np.zeros((1,) + env.observation_space.shape, dtype = np.float32))
target_model.set_weights(model.get_weights())


loss_fn = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate = 1e-4)

replay_buffer = deque(maxlen = REPLAY_CAPACITY)
replay_array = np.empty() # El objetivo es usar el indexado de C que nos ofrece numpy

all_rewards, all_losses = [], []
train_steps = 0

start = datetime.now()
last_time = start
for episode in range(EPISODES):
    current_state, info = env.reset()

    total_reward = 0
    losses = []

    epsilon = max(1 - episode / (EPISODES * 0.9), 0.01) if not SAVED_MODEL else 0.01

    for step in range(MAX_EPISODE_STEPS):
        action = epsilon_greedy_policy(current_state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)

        replay_buffer.append((current_state, action, reward, next_state, terminated or truncated))
        total_reward += reward
        current_state = next_state

        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = sample_experiences(BATCH_SIZE)            
            next_Q_values = model(next_states, training=False)
            best_next_actions = np.argmax(next_Q_values, axis = 1)

            next_mask = tf.one_hot(best_next_actions, env.action_space.n)
            next_best_Q_values = tf.reduce_sum((target_model(next_states, training = False) * next_mask), axis = 1)

            target_Q_values = (rewards + (1 - dones) * DISCOUNT_FACTOR * next_best_Q_values)
            target_Q_values = target_Q_values[:, np.newaxis]

            mask = tf.one_hot(actions, env.action_space.n)

            with tf.GradientTape() as tape:
                all_Q_values = model(states, training = True)
                Q_values = tf.reduce_sum(all_Q_values * mask, axis = 1, keepdims = True)
                loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            losses.append(loss.numpy())
            train_steps += 1

            if train_steps % TARGET_UPDATE_EVERY_STEPS == 0:
                target_model.set_weights(model.get_weights())


        if terminated or truncated:
            print(f"Terminado {terminated}   Truncado {truncated}")
            break

    all_rewards.append(total_reward)
    if losses:
        all_losses.append(np.mean(losses))

    if episode % INFO_EACH_EPISODES == 0:
        mean_reward = np.mean(all_rewards[-10:])
        mean_loss = np.mean(all_losses[-10:]) if all_losses else 0
        
        print(f"\nEpisodio {episode:3d}: Recompensa media = {mean_reward:.1f}\npérdida media ≈ {mean_loss:.4f}\nTiempo: {datetime.now() - last_time if last_time else None}")
        last_time = datetime.now()
        

env.close()

print("Fin del programa, Tiempo: ", datetime.now() - start)

model.save(MODEL_PATH)

plt.plot(all_rewards)
plt.xlabel("Episodios")
plt.ylabel("Recompensa total")
plt.title("Evolución de la recompensa")
plt.show()
