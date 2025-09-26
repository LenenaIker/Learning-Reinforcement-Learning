
import gymnasium as gym
import keras
import numpy as np
from collections import deque
import tensorflow as tf
tf.get_logger().setLevel("ERROR") # Pa que se calle el pesau
from matplotlib import pyplot as plt
import os
from datetime import datetime

MODEL_PATH = "neKabuz/f06Model.h5"


modelo_disponible = os.path.exists(MODEL_PATH)
print("Modelo enocntrado: ", modelo_disponible)

EPISODES = 500 if not modelo_disponible else 5
STEPS_PER_EPISODE = 200



env = gym.make(
    "MountainCar-v0",
    render_mode = "human" if modelo_disponible else None,
    max_episode_steps = STEPS_PER_EPISODE
)



if modelo_disponible:
    model = keras.models.load_model(MODEL_PATH)
    
else:
    model = keras.models.Sequential([
        keras.Input(shape = env.observation_space.shape),
        keras.layers.Dense(32, activation = "elu"),
        keras.layers.Dense(32, activation = "elu"),
        keras.layers.Dense(env.action_space.n)
    ])



# Función para normalizar a escala [-1, 1] las observaciones proporcionadas por env
obs_low  = env.observation_space.low.astype(np.float32)
obs_high = env.observation_space.high.astype(np.float32)

def normalize(state):
    scale = np.where((obs_high - obs_low) == 0, 1.0, (obs_high - obs_low))
    return ((state - obs_low) / scale * 2.0 - 1.0).astype(np.float32)


def epsilon_greedy_policy(state, epsilon = 0):

    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    else:
        Q_values = model(state[np.newaxis], training = False).numpy()
        return np.argmax(Q_values[0])



def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size = batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, terminateds, truncateds = [
        np.array([ experience[field_index] for experience in batch ]) for field_index in range(6)
    ]
    return states, actions, rewards, next_states, terminateds, truncateds




batch_size = 32
replay_buffer = deque(maxlen = 2000)

all_rewards = []
all_losses = []

discount_factor = 0.95

loss_fn = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate = 1e-3)


start = datetime.now()
last_time = start
for episode in range(EPISODES):
    current_state, info = env.reset()
    current_state = normalize(current_state)

    total_reward = 0
    losses = []

    epsilon = max(1 - episode / (EPISODES * 0.9), 0.05) if not modelo_disponible else 0.01

    for step in range(STEPS_PER_EPISODE - 1):
        action = epsilon_greedy_policy(state = current_state, epsilon = epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = normalize(next_state)
        replay_buffer.append((current_state, action, reward, next_state, terminated, truncated))
        total_reward += reward

        current_state = next_state

        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, terminateds, truncateds = sample_experiences(batch_size)
            dones = np.logical_or(terminateds, truncateds).astype(np.float32)
            next_Q_values = model.predict(next_states, verbose = 0)

            max_next_Q_values = np.max(next_Q_values, axis = 1)
            target_Q_values = (rewards + (1 - dones) * discount_factor * max_next_Q_values)
            target_Q_values = target_Q_values[:, np.newaxis] # Hau gabe ondo doa, Kerasek be kabuz iteulako. Halatare hola egonkorragoa dala dio Gepetok.

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

    if episode % 3 == 0:
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
