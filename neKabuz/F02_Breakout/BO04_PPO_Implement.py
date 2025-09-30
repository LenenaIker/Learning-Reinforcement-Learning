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
from RolloutBuffer import RolloutBuffer # Ne objetua


Z_PARTIDA = 500
Z_INTERAKZIO_PARTIDAKO = 5000

ZENBATEO_EGUNERATU_KRITIKOA = 2000

ROLLOUT_LENGTH = 256

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 1e-4



gym.register_envs(ale_py)
env = gym.make(
    "ALE/Breakout-v5",
    max_episode_steps = Z_INTERAKZIO_PARTIDAKO + 1
)


actor = keras.models.Sequential([
    keras.Input(shape = env.observation_space.shape),
    keras.layers.Lambda(lambda t: tf.cast(t, tf.float32) / 255.0),
    keras.layers.Conv2D(32, 8, strides = 4, activation = "relu"),
    keras.layers.Conv2D(64, 4, strides = 2, activation = "relu"),
    keras.layers.Conv2D(64, 3, strides = 1, activation = "relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation = "elu"),
    keras.layers.Dense(256, activation = "elu"),
    keras.layers.Dense(env.action_space.n, activation = None) # Cambio de planes. Ver: neKabuz\F02_Breakout\LogProbVSlog_softmaxLogits.py
])


critic = keras.models.Sequential([
    keras.Input(shape = env.observation_space.shape),
    keras.layers.Lambda(lambda t: tf.cast(t, tf.float32) / 255.0),
    keras.layers.Conv2D(32, 8, strides = 4, activation = "relu"),
    keras.layers.Conv2D(64, 4, strides = 2, activation = "relu"),
    keras.layers.Conv2D(64, 3, strides = 1, activation = "relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation = "elu"),
    keras.layers.Dense(256, activation = "elu"),
    keras.layers.Dense(1, activation = None)
])




for partida in range(Z_PARTIDA):
    trajectories = RolloutBuffer(ROLLOUT_LENGTH, env.observation_space.shape)
    
    egoera_oain, info = env.reset() # Partida hasi

    lives = info.get("lives")
    for interakzio in range(Z_INTERAKZIO_PARTIDAKO):
        action_logits = actor(egoera_oain[np.newaxis], training = False) # Aktoreak akzio bakoitzan logitak itzultzeitu.
        
        # Muestreo estocastico | muestreo categórico
        action = tf.random.categorical(action_logits, num_samples = 1) # Honek bakoitzan prob-ak kontuan izanda, gambleatu iteu ze akzio erabakitzeko. Gambleo honek explorazioan aportatzeu
        logprobs = tf.nn.log_softmax(action_logits)

        value = critic(egoera_oain[np.newaxis], training = False) # Kritikoak be balorazioa iteu.

        egoera_gero, reward, terminated, truncated, info = env.step(action) # Ingurumenai aktoreak erabakitako akzioa pasateiou, ta honek ondoriozko emaitzak pasatzeizkigu


        if np.array_equal(egoera_oain, egoera_gero):
            reward -= 1 # Pelota sakatze eztun bitartian penalizatu.

        if lives > info.get("lives"):
            reward -= 10 # Bizitza galtzeunean zigortu
            lives = info.get("lives")


        trajectories.store(
            state = egoera_oain,
            action = action,
            reward = reward,
            terminated = terminated,
            truncated = truncated,
            logprob = logprobs[action],
            value = value
        )

        egoera_oain = egoera_gero

        if terminated or truncated:
            print(f"\nINFO: Terminated {terminated} | Truncated {truncated}\n")
            break
        







env.close()

