import gymnasium as gym
import ale_py

from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation

from gymnasium.vector import AsyncVectorEnv
import multiprocessing as mp

import keras
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import numpy as np
from datetime import datetime

from RolloutBufferBatched import RolloutBufferBatched


MODEL_PATH = "neKabuz/F02_Breakout/BO04_actor_critic.keras"

Z_PARTIDA = 500
Z_INTERAKZIO_PARTIDAKO = 5000

ROLLOUT_LENGTH = 1024 

DISCOUNT_FACTOR = tf.constant(0.99, tf.float32)
LEARNING_RATE = 1e-4
GAE_LAMBDA = tf.constant(0.95, tf.float32)
CLIP_RANGE = 0.20
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
TRAIN_EPOCHS = 4
MINIBATCH_SIZE = 256

N_ENVS = 8
SCREEN_SIZE = 84
FRAME_SKIP = 4
FRAME_STACK = 4

INPUT_SHAPE = None
N_ACTIONS = None


actor_critic = None
optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
huber = keras.losses.Huber(delta = 1.0) 



def make_env(seed: int):
    def thunk():
        gym.register_envs(ale_py)
        env = gym.make(
            "ALE/Breakout-v5",
            max_episode_steps = Z_INTERAKZIO_PARTIDAKO + 1,
            frameskip = 1
        )
        env = AtariPreprocessing(
            env,
            screen_size = SCREEN_SIZE,
            grayscale_obs = True,
            frame_skip = FRAME_SKIP,
            noop_max = 30
        )
        env = FrameStackObservation(env, stack_size = FRAME_STACK)
        # Al usar FrameStack, el shape cambia, hay que adaptarlo con lo de abajo:
        env = TransformObservation(
            env = env,
            func = lambda obs: tf.transpose(obs, (1, 2, 0)),
            observation_space = gym.spaces.Box(
                low = 0,
                high = 255,
                shape = (SCREEN_SIZE, SCREEN_SIZE, FRAME_STACK),
                dtype = np.uint8
            )
        )
        env.reset(seed = seed)
        return env
    return thunk

def build_env():
    # fuerza 'spawn' para máxima compatibilidad (Windows/macOS/notebooks)
    return AsyncVectorEnv(
        [make_env(seed = i) for i in range(N_ENVS)],
        context = "spawn"
    )



def bifidModel():
    inputs = keras.Input(shape = INPUT_SHAPE, dtype = tf.uint8)

    x = keras.layers.Rescaling(scale = 1.0 / 255.0, dtype = "float32")(inputs)

    x = keras.layers.Conv2D(32, 8, strides = 4, activation = "relu")(x)
    x = keras.layers.Conv2D(64, 4, strides = 2, activation = "relu")(x)
    x = keras.layers.Conv2D(64, 3, strides = 1, activation = "relu")(x)

    x = keras.layers.Flatten()(x)

    h = keras.layers.Dense(512, activation = "relu")(x)
    
    policy_logits = keras.layers.Dense(N_ACTIONS, activation = None, name = "policy_logits")(h)
    value = keras.layers.Dense(1, activation = None, name = "value")(h)

    return keras.Model(inputs = inputs, outputs = [policy_logits, value])

     

def categorical_entropy_from_logits(logits: tf.Tensor) -> tf.Tensor:
    """
    Entropía media de la política categórica a partir de logits.
    """
    log_probs = tf.nn.log_softmax(logits, axis = -1)
    probs = tf.nn.softmax(logits, axis = -1)
    entropy = - tf.reduce_sum(probs * log_probs, axis = -1)
    return tf.reduce_mean(entropy)


@tf.function(jit_compile = True)
def compute_gae_tf_batched(rewards, values, dones, last_values, gamma = DISCOUNT_FACTOR, lam = GAE_LAMBDA):
    # rewards, values, dones: (T, N) ; last_values: (N,)
    not_dones = 1. - tf.cast(dones, tf.float32)                           # (T, N)
    v_next = tf.concat([values[1:], last_values[tf.newaxis, :]], axis = 0)  # (T, N)
    deltas = rewards + gamma * v_next * not_dones - values                # (T, N)

    def scan_fn(carry, elems):
        delta, nd = elems  # (N,), (N,)
        carry = delta + gamma * lam * nd * carry
        return carry

    # escaneo hacia atrás por la dimensión T
    advantages = tf.scan(
        fn = scan_fn,
        elems = (tf.reverse(deltas, [0]), tf.reverse(not_dones, [0])),
        initializer = tf.zeros_like(deltas[0]),
    )
    advantages = tf.reverse(advantages, [0])  # (T, N)
    returns = advantages + values

    # normalización por el lote completo (T*N)
    flat_adv = tf.reshape(advantages, [-1])
    adv_mean, adv_var = tf.nn.moments(flat_adv, axes = [0])
    advantages = (advantages - adv_mean) / tf.sqrt(adv_var + 1e-8)

    return advantages, returns




def _ppo_minibatch_step(states_mb, actions_mb, old_logprobs_mb, advantages_mb, returns_mb):
    """
    Paso de optimización PPO sobre un minibatch, usando:
      - sparse_softmax_cross_entropy -> log-probs
      - Huber loss para el crítico
    """
    with tf.GradientTape(persistent = True) as tape:

        logits, values_pred = actor_critic(states_mb, training = True) 
        
        new_logprobs = - tf.nn.sparse_softmax_cross_entropy_with_logits(labels = actions_mb, logits = logits) 

        ratio = tf.exp(new_logprobs - old_logprobs_mb) 

        unclipped = ratio * advantages_mb
        clipped = tf.clip_by_value(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * advantages_mb
        policy_loss = - tf.reduce_mean(tf.minimum(unclipped, clipped))

        entropy_bonus = categorical_entropy_from_logits(logits)
        value_loss = huber(returns_mb, tf.squeeze(values_pred, axis = -1))

        total_loss = (policy_loss - ENTROPY_COEF * entropy_bonus + VALUE_COEF * value_loss)

    actor_grads = tape.gradient(total_loss, actor_critic.trainable_variables)
    del tape

    optimizer.apply_gradients(zip(actor_grads, actor_critic.trainable_variables))

    
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



def train():
    global INPUT_SHAPE, N_ACTIONS, actor_critic

    env = build_env()
    if not INPUT_SHAPE or not N_ACTIONS or not actor_critic:
        INPUT_SHAPE = env.single_observation_space.shape
        N_ACTIONS = env.single_action_space.n        
        actor_critic = bifidModel()


    rollout = RolloutBufferBatched(ROLLOUT_LENGTH, INPUT_SHAPE, N_ENVS)
    start = datetime.now()
    for partida in range(Z_PARTIDA):
        obs, infos = env.reset()

        for interakzio in range(Z_INTERAKZIO_PARTIDAKO):
            logits, values = actor_critic(tf.convert_to_tensor(obs, tf.uint8), training = False)
            
            actions = tf.squeeze(tf.random.categorical(
                logits,
                num_samples = 1,
                dtype = tf.int32
            ), axis = -1)
            
            log_probs = tf.nn.log_softmax(logits, axis = -1)

            idx = tf.stack([tf.range(N_ENVS, dtype = tf.int32), actions], axis = 1)
            logprob_action = tf.gather_nd(log_probs, idx)

            next_obs, rewards, terminateds, truncateds, infos = env.step(actions.numpy()) 
            dones = np.logical_or(terminateds, truncateds)

            rollout.store_batch(
                state = obs,
                action = actions.numpy(),
                reward = rewards,
                done = dones,
                logprob = logprob_action.numpy(),
                value = tf.squeeze(values, axis = -1).numpy()
            )

            obs = next_obs

            if rollout.is_full():
                _, last_val = actor_critic(tf.convert_to_tensor(next_obs, tf.uint8), training = False) # (N, 1)
                last_val = tf.squeeze(last_val, axis = -1) # (N, )

                states, actions_b, rewards_b, dones_b, logprobs_b, values_b = rollout.getXPs()
            
                advantages, returns = compute_gae_tf_batched(
                    tf.convert_to_tensor(rewards_b, tf.float32),
                    tf.convert_to_tensor(values_b, tf.float32),
                    tf.convert_to_tensor(dones_b, tf.bool),
                    last_val
                )
                
                T = states.shape[0]
                states_tf = tf.reshape(tf.convert_to_tensor(states, tf.uint8), (T * N_ENVS,) + INPUT_SHAPE)
                actions_tf = tf.reshape(tf.convert_to_tensor(actions_b, tf.int32), (T * N_ENVS,))
                oldlp_tf = tf.reshape(tf.convert_to_tensor(logprobs_b, tf.float32), (T * N_ENVS,))
                adv_tf = tf.reshape(advantages, (T * N_ENVS,))
                ret_tf = tf.reshape(returns, (T * N_ENVS,))

                ppo_update_tf(states_tf, actions_tf, oldlp_tf, adv_tf, ret_tf)

                rollout.reset()

            if interakzio % 250 == 0:
                print("\nPartida: ", partida, " / ", Z_PARTIDA, "\nStep: ", interakzio, " / ", Z_INTERAKZIO_PARTIDAKO, "\nTime: ", datetime.now() - start)
    actor_critic.save(MODEL_PATH)

    env.close()

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force = True)
    except RuntimeError as e:
        print(e)

    # if .exe with PyInstaller:
    # mp.freeze_support()
    train()
