
import gymnasium as gym
import numpy as np
import keras
from collections import deque
import tensorflow as tf

from matplotlib import pyplot as plt



env = gym.make("CartPole-v0") # , render_mode = "human")


model = keras.models.Sequential([
    keras.Input(shape = env.observation_space.shape),
    keras.layers.Dense(32, activation = "elu"),
    keras.layers.Dense(32, activation = "elu"),
    keras.layers.Dense(env.action_space.n)
])

def epsilon_greedy_policy(state, epsilon = 0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis], verbose = 0)
        return np.argmax(Q_values[0])


replay_buffer = deque(maxlen = 2000)


def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size = batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, terminateds, truncateds = [
        np.array([ experience[field_index] for experience in batch ]) for field_index in range(6) # Liburuan 5 jartzeu baino done oain terminated ta truncated dia.
    ]

    return states, actions, rewards, next_states, terminateds, truncateds

def play_one_step(env: gym.Env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, terminated, truncated, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, terminated, truncated))

    return next_state, reward, terminated, truncated, info


batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(learning_rate = 1e-3)
loss_fn = keras.losses.mean_squared_error

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, terminateds, truncateds = experiences
    dones = np.logical_or(terminateds, truncateds).astype(np.float32)
    next_Q_values = model.predict(next_states, verbose = 0)
    max_next_Q_values = np.max(next_Q_values, axis = 1)
    target_Q_values = (
        rewards + (1 - dones) * discount_factor * max_next_Q_values
    )
    mask = tf.one_hot(actions, env.action_space.n) # env.action_space.n == nn.output.size
    
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis = 1, keepdims = True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss.numpy()



all_rewards = []
all_losses = []

for episode in range(600):
    obs, _ = env.reset()
    total_reward = 0
    losses = []

    epsilon = max(1 - episode / 500, 0.01)
    for step in range(200):
        obs, reward, terminated, truncated, info = play_one_step(env, obs, epsilon)
        total_reward += reward

        if terminated or truncated:
            break

    if episode > 50:
        loss = training_step(batch_size)
        losses.append(loss)

    all_rewards.append(total_reward)
    if losses:
        all_losses.append(np.mean(losses))


    if episode % 10 == 0:
        mean_reward = np.mean(all_rewards[-10:])
        mean_loss = np.mean(all_losses[-10:]) if all_losses else 0
        print(f"Episodio {episode:3d}: Recompensa media = {mean_reward:.1f}, "
              f"ε = {epsilon:.2f}, pérdida media ≈ {mean_loss:.4f}")

env.close()



plt.plot(all_rewards)
plt.xlabel("Episodios")
plt.ylabel("Recompensa total")
plt.title("Evolución de la recompensa en CartPole")
plt.show()
