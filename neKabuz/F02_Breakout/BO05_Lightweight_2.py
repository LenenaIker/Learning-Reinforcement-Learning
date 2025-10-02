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


MODEL_PATH = "neKabuz/F02_Breakout/BO04_actor_critic.keras"

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

INPUT_SHAPE = (84, 84, 1) # grayscalek canalak gutxitzeitu 3 tik 1ea. resizek 84x84.
N_ACTIONS = env.action_space.n


# Preprocesado de imágen fuera del modelo. Así ahorro a la hora de guardar en el buffer y a la hora de entrenar. (forward más ligero)
# Imágen a escala de grisies, reducir resolución a 84 * 84
def preprocess(obs):
    x = tf.image.rgb_to_grayscale(obs)
    x = tf.image.resize(x, [84,84], method = 'area')
    return tf.cast(x, tf.uint8)


def bifidModel():
    inputs = keras.Input(shape = INPUT_SHAPE, dtype = tf.uint8)

    # Normalizar a [0, 1]
    x = keras.layers.Rescaling(scale = 1./255, dtype = 'float32')(inputs)

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

optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
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


@tf.function(jit_compile = True)
def compute_gae_tf(rewards, values, dones, last_value, gamma = DISCOUNT_FACTOR, lam = GAE_LAMBDA):
    # rewards, values, dones: [T] float32 / bool
    not_dones = 1. - tf.cast(dones, tf.float32)
    v_next = tf.concat([values[1:], tf.reshape(tf.cast(last_value, tf.float32), [1])], 0)
    deltas = rewards + gamma * v_next * not_dones - values

    def scan_fn(gae, elems):
        delta, nd = elems
        gae = delta + gamma * lam * nd * gae
        return gae

    advantages = tf.scan(
        fn = scan_fn,
        elems = (tf.reverse(deltas, [0]), tf.reverse(not_dones, [0])),
        initializer = tf.zeros((), tf.float32),
    )
    advantages = tf.reverse(advantages, [0])
    returns = advantages + values

    adv_mean, adv_var = tf.nn.moments(advantages, axes = [0])
    advantages = (advantages - adv_mean) / tf.sqrt(adv_var + 1e-8)
    return advantages, returns




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

        total_loss = (policy_loss - ENTROPY_COEF * entropy_bonus + VALUE_COEF * value_loss)

    actor_grads = tape.gradient(total_loss, actor_critic.trainable_variables)
    del tape

    optimizer.apply_gradients(zip(actor_grads, actor_critic.trainable_variables))

    # Métricas útiles
    approx_kl = tf.reduce_mean(old_logprobs_mb - new_logprobs)
    clipfrac = tf.reduce_mean(tf.cast(tf.greater(
        tf.abs(ratio - 1.0),
        CLIP_RANGE
    ), tf.float32))

    return policy_loss, value_loss, entropy_bonus, approx_kl, clipfrac


def make_ds(states, actions, old_logprobs, advantages, returns, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((
        states, actions, old_logprobs, advantages, returns
    ))
    return ds.shuffle(8192, reshuffle_each_iteration = True).batch(batch_size, drop_remainder = True)


def ppo_update_tf(states, actions, old_logprobs, advantages, returns, epochs = TRAIN_EPOCHS, bs = MINIBATCH_SIZE):
    ds = make_ds(states, actions, old_logprobs, advantages, returns, bs)
    for _ in tf.range(epochs):
        for states_mb, actions_mb, old_lp_mb, adv_mb, ret_mb in ds:
            _ = _ppo_minibatch_step(states_mb, actions_mb, old_lp_mb, adv_mb, ret_mb)


for partida in range(Z_PARTIDA):
    trajectories = RolloutBuffer(ROLLOUT_LENGTH, INPUT_SHAPE)
    
    egoera_oain, info = env.reset()
    egoera_oain = preprocess(egoera_oain).numpy()

    lives = info.get("lives")
    for interakzio in range(Z_INTERAKZIO_PARTIDAKO):
        action_logits, value = actor_critic(egoera_oain[np.newaxis], training = False)
        
        # Muestreo estocastico | muestreo categórico
        action = tf.random.categorical(action_logits, num_samples = 1, dtype = tf.int32) # Honek bakoitzan prob-ak kontuan izanda, gambleatu iteu ze akzio erabakitzeko. Gambleo honek explorazioan aportatzeu
        action = tf.squeeze(action, axis = -1) # Reduce dimensiones, en este caso quita la última
        action = int(action[0])

        # Akzioen probabilitateen deribatua
        log_probs = tf.nn.log_softmax(action_logits)
        # Akzioan probabilitatean deribatua tensoretik atea
        logprob_action = tf.gather(log_probs[0], action)

        egoera_gero, reward, terminated, truncated, info = env.step(action) # Ingurumenai aktoreak erabakitako akzioa pasateiou, ta honek ondoriozko emaitzak pasatzeizkigu
        done = terminated or truncated
        egoera_gero = preprocess(egoera_gero).numpy()

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
            logprob = float(logprob_action),
            value = float(tf.squeeze(value))
        )


        if trajectories.is_full() or done:
            if done:
                last_val = tf.zeros((), tf.float32)
            else:
                _, last_val = actor_critic(egoera_gero[tf.newaxis, ...], training = False)
                last_val = tf.squeeze(last_val)

            states, actions, rewards, dones, logprobs, values = trajectories.getXPs()
        
            advantages, returns = compute_gae_tf(
                tf.convert_to_tensor(rewards, tf.float32),
                tf.convert_to_tensor(values, tf.float32),
                tf.convert_to_tensor(dones, tf.bool),
                last_val
            )
            
            ppo_update_tf(
                tf.convert_to_tensor(states, tf.uint8),
                tf.convert_to_tensor(actions, tf.int32),
                tf.convert_to_tensor(logprobs, tf.float32),
                advantages,
                returns
            )

            trajectories.reset()

        egoera_oain = egoera_gero

        if interakzio % 250 == 0:
            print("\nPartida: ", partida, " / ", Z_PARTIDA, "\nStep: ", interakzio, " / ", Z_INTERAKZIO_PARTIDAKO)
actor_critic.save(MODEL_PATH)

env.close()


# Partida bat bi minututan jolastu du honek.
# 2 min * 500 partida = 1000 minutu
# 1000 / 60 = 16.66 h

# Ta hoi, kontuan izan gabe agian geroz ta gehio iraungoutela partidek
# (eztet uste MountainCarren bezain beste haunditukoanik, ze rolloutBuffer berridazten doa, ezta tamainaz haunditzen.)

# Halatare denboa asko da. Bai o bai wrapperrak edo TF-Agents liburutegia erabili beharkot.
# Edo RunPot-en bota entrenatzeko.

