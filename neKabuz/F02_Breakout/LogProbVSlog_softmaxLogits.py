import tensorflow as tf

logits = tf.constant([1000.0, 0.0]) # valores extremos

# Opción 1: softmax + log
probs = tf.nn.softmax(logits)
log_probs_via_softmax = tf.math.log(probs)

# Opción 2: log_softmax
log_probs_direct = tf.nn.log_softmax(logits)

print("log(probs):", log_probs_via_softmax.numpy())
print("log_softmax(logits):", log_probs_direct.numpy())
