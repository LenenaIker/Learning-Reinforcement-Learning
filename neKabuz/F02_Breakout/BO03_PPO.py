# Interneten beidatuz PPO implementatzeko 2 aukera didtutelan konklusiora iritsi naiz.
#   1. Bi modelo definitu, Actor ta Critic. Aktoreak ekintzak erabakitzeitu ta Kritikoak, Aktorean erabakiak baloratukoitu (nota eman)
#
#   2. Modelo bifido bat definitu
#       Gorputza berdina izangoa, baino azken layerra ezta layer 1 izango, 2 layer independiente izangoia. Goikoan berdina iteutenak
#   
#   Gepetok esateu bigarren aukera merkeagoa dala entrenatzeko, baño ne ustez lehen aukera hobeto ulertzea.

import gymnasium as gym
import ale_py

import keras
import tensorflow as tf
tf.get_logger().setLevel("ERROR")


import numpy as np

EPISODES = 500
MAX_EPISODE_STEPS = 5000

TARGET_UPDATE_EVERY_STEPS = 2000

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 1e-4


gym.register_envs(ale_py)
env = gym.make(
    "ALE/Breakout-v5",
    max_episode_steps = MAX_EPISODE_STEPS + 1
)

# Dena ingouna
actor = keras.models.Sequential([
    keras.Input(shape = env.observation_space.shape),
    keras.layers.Lambda(lambda t: tf.cast(t, tf.float32) / 255.0),
    keras.layers.Conv2D(32, 8, strides = 4, activation = "relu"),
    keras.layers.Conv2D(64, 4, strides = 2, activation = "relu"),
    keras.layers.Conv2D(64, 3, strides = 1, activation = "relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation = "elu"),
    keras.layers.Dense(256, activation = "elu"),
    keras.layers.Dense(env.action_space.n, activation = 'softmax') # GOATU: Akzio bakoitzai probabilitate bat emateio
])


# Aktorea baloratukouna
critic = keras.models.Sequential([
    keras.Input(shape = env.observation_space.shape),
    keras.layers.Lambda(lambda t: tf.cast(t, tf.float32) / 255.0),
    keras.layers.Conv2D(32, 8, strides = 4, activation = "relu"),
    keras.layers.Conv2D(64, 4, strides = 2, activation = "relu"),
    keras.layers.Conv2D(64, 3, strides = 1, activation = "relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation = "elu"),
    keras.layers.Dense(256, activation = "elu"),
    keras.layers.Dense(1, activation = None) # <-- Output escalar, Hau izangoa aktorean nota
])

obs, info = env.reset()
try:
    print("Actor:    ", actor.predict(obs[np.newaxis], verbose = False))
    print("Critic:   ", critic.predict(obs[np.newaxis], verbose = False))
except Exception as e:
    print(e)

env.close()


# Honen emaitza hurrengo hau izan da:
# Actor:     [[0.25100312 0.25060597 0.24529074 0.25310013]]
# Critic:    [[0.09567405]] 

# Ikustean bezala softmaxen SUM(balore guztik) == 1 