import tensorflow as tf
import numpy as np

# following source codes are taken from:
# https://www.programcreek.com/python/example/90479/tensorflow.erf

def gelu(x):
  return tf.mul(x, tf.erfc(-x / tf.sqrt(2.)) / 2.)

# basically, we use fast_gelu
def fast_gelu(x):
  return 0.5 * x * (
    1 + tf.tanh(
      tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
    )
  )

