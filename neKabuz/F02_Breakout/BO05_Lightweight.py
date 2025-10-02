# Interneten beidatuz PPO implementatzeko 2 aukera didtutelan konklusiora iritsi naiz.
#   1. Bi modelo definitu, Actor ta Critic. Aktoreak ekintzak erabakitzeitu ta Kritikoak, Aktorean erabakiak baloratukoitu (nota eman)
#
#   2. Modelo bifido bat definitu
#       Gorputza berdina izangoa, baino azken layerra ezta layer 1 izango, 2 layer independiente izangoia. Goikoan berdina iteutenak
#   
#   Gepetok esateu bigarren aukera merkeagoa dala entrenatzeko, baño ne ustez lehen aukera hobeto ulertzea.

import gymnasium as gym
import ale_py

# Oaindik azkarrao nahi baet, hauek implementatu beharkoitut:
# from gymnasium.wrappers import AtariPreprocessing, FrameStack
# from gymnasium.vector import AsyncVectorEnv

import keras
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import numpy as np

# NeKabuz
from RolloutBuffer import RolloutBuffer


MODEL_PATH = "neKabuz\F02_Breakout\BO04_actor_critic.keras"

Z_PARTIDA = 500
Z_INTERAKZIO_PARTIDAKO = 5000

ROLLOUT_LENGTH = 1024 # He aumentado el tamaño para que se entrene menos veces, acelerando el proceso.

DISCOUNT_FACTOR = tf.constant(0.99, tf.float32)
LEARNING_RATE = 1e-4
GAE_LAMBDA = tf.constant(0.95, tf.float32)
CLIP_RANGE = 0.20
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
TRAIN_EPOCHS = 4
MINIBATCH_SIZE = 256


gym.register_envs(ale_py)
env = gym.make(
    "ALE/Breakout-v5",
    max_episode_steps = Z_INTERAKZIO_PARTIDAKO + 1
)

INPUT_SHAPE = env.observation_space.shape
N_ACTIONS = env.action_space.n


def bifidModel():
    inputs = keras.Input(shape = INPUT_SHAPE)

    # Imágen a escala de grisies, reducir resolución a 84 * 84 y normalizar a [0, 1]
    x = keras.layers.Lambda(lambda t: tf.image.resize(
        tf.image.rgb_to_grayscale(
            tf.cast(t, tf.float32)
        ),
        [84, 84]
    ) / 255.0)(inputs)

    x = keras.layers.Conv2D(32, 8, strides = 4, activation = "relu")(x)
    x = keras.layers.Conv2D(64, 4, strides = 2, activation = "relu")(x)
    x = keras.layers.Conv2D(64, 3, strides = 1, activation = "relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation = "elu")(x)
    h = keras.layers.Dense(256, activation = "elu")(x)

    # Dos cabezas, ambas heredan de h, que es el resto del modelo. Ambos devuelven Logits
    policy_logits = keras.layers.Dense(N_ACTIONS, activation = None, name = "policy_logits")(h)
    value = keras.layers.Dense(1, activation = None, name = "value")(h)

    return keras.Model(inputs = inputs, outputs = [policy_logits, value])

     
actor_critic = bifidModel()

actor_optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
critic_optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
huber = keras.losses.Huber(delta = 1.0) 

# === Utilidades PPO/GAE ===
def categorical_entropy_from_logits(logits: tf.Tensor) -> tf.Tensor:
    """
    Entropía media de la política categórica a partir de logits.
    """
    log_probs = tf.nn.log_softmax(logits, axis = -1)
    probs = tf.nn.softmax(logits, axis = -1)
    entropy = - tf.reduce_sum(probs * log_probs, axis = -1)
    return tf.reduce_mean(entropy)


def compute_gae(rewards, values, dones, last_value, gamma = DISCOUNT_FACTOR, lam = GAE_LAMBDA):
    """
    rewards: [T]
    values:  [T] (valores V(s_t) que guardaste en el buffer)
    dones:   [T] (bool/0-1)
    last_value: escalar V(s_{T}) para bootstrap si el episodio NO terminó, o 0.0 si done
    Devuelve:
      advantages: [T]
      returns:    [T] con R_t = A_t + V(s_t)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype = np.float32)
    not_dones = 1.0 - dones

    # v_{t+1} (shift) con bootstrap en el último
    v_next = tf.concat([
        values[1:],
        tf.reshape(tf.cast(last_value, tf.float32), [1])
    ], axis = 0)


    deltas = rewards + gamma * v_next * not_dones - values

    adv = tf.TensorArray(tf.float32, size = T)
    gae = tf.constant(0.0, tf.float32)

    gae = 0.0
    for t in range(T - 1, -1, -1):
        gae = deltas[t] + gamma * lam * not_dones[t] * gae
        adv = adv.write(t, gae)

    advantages = adv.stack()
    returns = advantages + values

    # Normalización de ventajas
    mean, var = tf.nn.moments(advantages, axes = [0])
    std = tf.sqrt(var + 1e-8)
    advantages = (advantages - mean) / std

    return tf.cast(advantages, tf.float32), tf.cast(returns, tf.float32)


@tf.function
def _ppo_minibatch_step(states_mb, actions_mb, old_logprobs_mb, advantages_mb, returns_mb):
    """
    Paso de optimización PPO sobre un minibatch, usando:
      - sparse_softmax_cross_entropy -> log-probs
      - Huber loss para el crítico
    """
    with tf.GradientTape(persistent = True) as tape:

        logits, values_pred = actor_critic(states_mb, training = True) # [B,A], [B,1]
        
        new_logprobs = - tf.nn.sparse_softmax_cross_entropy_with_logits(labels = actions_mb, logits = logits) # [B]

        ratio = tf.exp(new_logprobs - old_logprobs_mb) # [B]

        unclipped = ratio * advantages_mb
        clipped = tf.clip_by_value(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * advantages_mb
        policy_loss = - tf.reduce_mean(tf.minimum(unclipped, clipped))

        entropy_bonus = categorical_entropy_from_logits(logits)
        value_loss = huber(returns_mb, values_pred)

        actor_loss = policy_loss - ENTROPY_COEF * entropy_bonus
        critic_loss = VALUE_COEF * value_loss

    actor_grads = tape.gradient(actor_loss, actor_critic.trainable_variables)
    critic_grads = tape.gradient(critic_loss, actor_critic.trainable_variables)
    del tape

    actor_optimizer.apply_gradients(zip(actor_grads, actor_critic.trainable_variables))
    critic_optimizer.apply_gradients(zip(critic_grads, actor_critic.trainable_variables))

    # Métricas útiles
    approx_kl = 0.5 * tf.reduce_mean(tf.square(new_logprobs - old_logprobs_mb))
    clipfrac = tf.reduce_mean(tf.cast(tf.greater(
        tf.abs(ratio - 1.0),
        CLIP_RANGE
    ), tf.float32))

    return policy_loss, value_loss, entropy_bonus, approx_kl, clipfrac


def ppo_update(states, actions, old_logprobs, advantages, returns, epochs = TRAIN_EPOCHS, minibatch_size = MINIBATCH_SIZE):
    """
    Baraja y actualiza en minibatches durante 'epochs'.
    """
    N = states.shape[0]
    idxs = np.arange(N)

    for _ in range(epochs):
        np.random.shuffle(idxs)
        for start in range(0, N, minibatch_size):
            end = start + minibatch_size
            mb_idx = idxs[start:end]

            states_mb = tf.convert_to_tensor(states[mb_idx], dtype = tf.uint8)  # tus obs son uint8
            actions_mb = tf.convert_to_tensor(actions[mb_idx], dtype = tf.int32)
            old_logprobs_mb = tf.convert_to_tensor(old_logprobs[mb_idx], dtype = tf.float32)
            advantages_mb = tf.convert_to_tensor(advantages[mb_idx], dtype = tf.float32)
            returns_mb = tf.convert_to_tensor(returns[mb_idx], dtype = tf.float32)

            _ = _ppo_minibatch_step(
                states_mb, actions_mb, old_logprobs_mb, advantages_mb, returns_mb
            )



for partida in range(Z_PARTIDA):
    trajectories = RolloutBuffer(ROLLOUT_LENGTH, env.observation_space.shape)
    
    egoera_oain, info = env.reset()

    lives = info.get("lives")
    for interakzio in range(Z_INTERAKZIO_PARTIDAKO):
        action_logits, value = actor_critic(egoera_oain[np.newaxis], training = False)
        
        # Akzioen probabilitateen deribatua
        log_probs = tf.nn.log_softmax(action_logits)


        # Muestreo estocastico | muestreo categórico
        action = tf.random.categorical(action_logits, num_samples = 1, dtype = tf.int32) # Honek bakoitzan prob-ak kontuan izanda, gambleatu iteu ze akzio erabakitzeko. Gambleo honek explorazioan aportatzeu
        action = tf.squeeze(action, axis = -1) # Reduce dimensiones, en este caso quita la última
        action = int(action.numpy()[0])

        logprob_action = tf.gather(log_probs[0], action)

        egoera_gero, reward, terminated, truncated, info = env.step(action) # Ingurumenai aktoreak erabakitako akzioa pasateiou, ta honek ondoriozko emaitzak pasatzeizkigu
        done = terminated or truncated


        # Oaingoz eztet reward shaping erabiliko
        # if np.array_equal(egoera_oain, egoera_gero):
        #     reward -= 1 # Pelota sakatze eztun bitartian penalizatu.
        # if lives > info.get("lives"):
        #     reward -= 10 # Bizitza galtzeunean zigortu
        #     lives = info.get("lives")


        trajectories.store(
            state = egoera_oain,
            action = action,
            reward = reward,
            done = done,
            logprob = float(logprob_action.numpy()),
            value = float(value.numpy().squeeze())
        )


        if trajectories.is_full() or done:
            if done:
                last_val = 0.0
            else:
                _, last_val = float(actor_critic(egoera_gero[np.newaxis], training = False).numpy().squeeze())

            states, actions, rewards, dones, logprobs, values = trajectories.getXPs()
        
            advantages, returns = compute_gae(rewards, values, dones, last_val)
            
            advantages = advantages.numpy()
            returns = returns.numpy()

            ppo_update(states, actions, logprobs, advantages, returns)

            trajectories.reset()

        
        egoera_oain = egoera_gero
    print("Partida: ", partida, " / ", Z_PARTIDA)

actor_critic.save(MODEL_PATH)

env.close()


# Gauzak geoz ta hobeto ulertzeitut, hoi bai. Código hau ezin det ne ordenagaiuan exekutatu. RunPot erabiltzeko intentzioak dazkat.
# Ia ordu bat eon da, ta eztu partida bat bukatu. 500 ditu jolasteko XD.
# Gepetok esateit Grayscale erabiltzeko ta modelo bifido bakarra definitzeko actor ta critic-en ordez.