import tensorflow as tf
import numpy as np

# following source codes are taken from:
# https://www.programcreek.com/python/example/90479/tensorflow.erf

def gelu(x):
  return tf.multiply(x, tf.erfc(-x / tf.sqrt(2.)) / 2.)

# basically, we use fast_gelu
def fast_gelu(x):
  return 0.5 * x * (
    1 + tf.tanh(
      tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
    )
  )

# in BERT implementation, it is the fast-gelu algorithm
# except the cdf part is extracted as a named node in the graph
def bert_gelu(x):
  cdf = 0.5 * (
    1.0 + tf.tanh((
      np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
    ))
  )
  return x * cdf

# Practically, these 2 functions can be replaced as approximation
# However, gelu != fast_gelu in fact, just approximation for fast execution

if __name__ == "__main__":
  tf.enable_eager_execution()
  x = tf.Variable(1.2)
  v1, v2 = gelu(x), fast_gelu(x)
  print(v1, v2)

