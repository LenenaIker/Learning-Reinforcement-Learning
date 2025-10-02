import tensorflow as tf

# Supongamos que tenemos un batch de log-probs de una política
# (batch_size=2, n_actions=4)
batch = [
    [-1.2, -0.7, -3.4, -2.0],   # log_probs del estado 1
    [-0.3, -2.1, -0.9, -1.5]    # log_probs del estado 2
]

log_probs = tf.constant(batch, dtype = tf.float32)

# Acciones tomadas por la política en cada estado
actions = tf.constant([1, 2], dtype=tf.int32)

# --- Método 1: tf.gather (por batch)
# gather_nd permite seleccionar (batch, action) directamente
indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis = 1)
chosen_logprobs_gather = tf.gather_nd(log_probs, indices)

# --- Método 2: one-hot * log_probs + reduce_sum
one_hot_actions = tf.one_hot(actions, depth=tf.shape(log_probs)[1])
chosen_logprobs_onehot = tf.reduce_sum(one_hot_actions * log_probs, axis=-1)

print("Log probs:\n", log_probs.numpy())
print("Acciones:\n", actions.numpy())
print("\nMétodo gather_nd:", chosen_logprobs_gather.numpy())
print("Método one_hot + reduce_sum:", chosen_logprobs_onehot.numpy())


# gather == reduce_sum(one_hot)

# gather: selecciona elementos a lo largo de un eje.
# gather_nd: selecciona elementos usando coordenadas completas (como un acceso directo x[i,j,...]).