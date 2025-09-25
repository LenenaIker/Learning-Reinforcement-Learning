
import gymnasium as gym
import keras
import numpy as np
from collections import deque
import tensorflow as tf
from matplotlib import pyplot as plt
import os
from datetime import datetime

MODEL_PATH = "neKabuz/f03Model.h5"


modelo_disponible = os.path.exists(MODEL_PATH)
print("Modelo enocntrado: ", modelo_disponible)

EPISODES = 100 if not modelo_disponible else 10 
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




def epsilon_greedy_policy(state, epsilon = 0):

    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n) 
    else:
        Q_values = model.predict(state[np.newaxis], verbose = 0)
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

loss_fn = keras.losses.mean_squared_error
optimizer = keras.optimizers.Adam(learning_rate = 1e-3)


last_time = datetime.now()

max_max_position = -1
max_max_speed = 0

for episode in range(EPISODES):
    current_state, info = env.reset()
    
    total_reward = 0
    losses = []

    
    epsilon = 1 - episode / (EPISODES + EPISODES * 0.1) if not modelo_disponible else 0.1

    max_position = -1
    max_speed = 0

    for step in range(STEPS_PER_EPISODE - 1):
        action = epsilon_greedy_policy(state = current_state, epsilon = epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)


        # Rewarda gehitu maximoak gainditzen baitu:
        if max_position < next_state[0]:
            reward += 1
            max_position = next_state[0]
            if max_max_position < next_state[0]:
                reward += 5
                max_max_position = next_state[0]

        if max_speed < abs(next_state[1]):
            reward += 1
            max_speed = abs(next_state[1])
            if max_max_speed < abs(next_state[1]):
                reward += 5
                max_max_speed = abs(next_state[1])


        replay_buffer.append((current_state, action, reward, next_state, terminated, truncated))
        total_reward += reward

        current_state = next_state

        if terminated or truncated:
            print(f"Terminado {terminated}   Truncado {truncated}")
            break

    if episode > 50:
        states, actions, rewards, next_states, terminateds, truncateds = sample_experiences(batch_size)
        dones = np.logical_or(terminateds, truncateds).astype(np.float32)
        next_Q_values = model.predict(next_states, verbose = 0)

        max_next_Q_values = np.max(next_Q_values, axis = 1)
        target_Q_values = (rewards + (1 - dones) * discount_factor * max_next_Q_values)

        mask = tf.one_hot(actions, env.action_space.n)

        with tf.GradientTape() as tape:
            all_Q_values = model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis = 1, keepdims = True)
            loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        losses.append(loss.numpy())
    
    all_rewards.append(total_reward)
    if losses:
        all_losses.append(np.mean(losses))

    if episode % 10 == 0:
        mean_reward = np.mean(all_rewards[-10:])
        mean_loss = np.mean(all_losses[-10:]) if all_losses else 0
        
        print(f"\nEpisodio {episode:3d}: Recompensa media = {mean_reward:.1f}\nε = {epsilon:.2f}, pérdida media ≈ {mean_loss:.4f}\nTiempo: {datetime.now() - last_time if last_time else None}")
        last_time = datetime.now()

env.close()


model.save(MODEL_PATH)


plt.plot(all_rewards)
plt.xlabel("Episodios")
plt.ylabel("Recompensa total")
plt.title("Evolución de la recompensa en CartPole")
plt.show()
