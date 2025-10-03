import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

import os


def play_one_step(env: gym.Env, obs, model, loss_fn):
    with tf.GradientTape() as tape: # Las variables se quedan "pegadas" al celo(tape), para luego calcular gradiente.
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32) # Tensorflow funciona con 32 bits, porque a la IA no le hacen falta 64
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))

    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, terminated, truncated, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, terminated, truncated, grads
    

def play_multiple_episodes(env: gym.Env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs, info = env.reset()
        
        for step in range(n_max_steps):
            obs, reward, terminated, truncated, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if terminated or truncated:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    
    return all_rewards, all_grads


def discount_rewards(rewards, discount_factor): # Se usa para que un reward afecte en los pasos anteriores, pero con un descuento.
    discounted = np.array(rewards)
    
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()

    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


if __name__ == "__main__":
        
    env = gym.make("CartPole-v1", render_mode = "human")
    MODEL_PATH = "f02_PoleModel.h5"

    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
    else:
        n_inputs = 4 # Layer de imputs de la NN, lo que vendría siendo cantidad de Xes en: y = x1 * w1 + x2 * w2 + ... + b

        # En este caso para hacerlo automático también podríamos poner
        # n_inputs = env.observation_space.shape[0]

        model = keras.models.Sequential([
            keras.Input(shape = (n_inputs, )),
            keras.layers.Dense(5, activation = "elu"), # input_shape = [n_inputs]),
            keras.layers.Dense(1, activation = "sigmoid")
        ])


    n_iterations = 5# 150
    n_episodes_per_update = 10
    n_max_steps = 200
    discount_factor = 0.95

    optimizer = keras.optimizers.Adam(learning_rate = 0.01)
    loss_fn = keras.losses.binary_crossentropy


    for iteration in range(n_iterations):
        all_rewards, all_grads = play_multiple_episodes(
            env = env,
            n_episodes = n_episodes_per_update,
            n_max_steps = n_max_steps,
            model = model,
            loss_fn = loss_fn
        )

        all_final_rewards = discount_and_normalize_rewards(
            all_rewards = all_rewards,
            discount_factor = discount_factor
        )

        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[episode_index][step][var_index] for episode_index, final_rewards in enumerate(all_final_rewards) for step, final_reward in enumerate(final_rewards)],
                axis = 0
            )

            all_mean_grads.append(mean_grads)

        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))



    # model.save(MODEL_PATH)


    env.close()
