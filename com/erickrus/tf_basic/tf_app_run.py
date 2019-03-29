# This file is to demostrate
# 1) how tf.app.run() works
# https://stackoverflow.com/questions/33703624/how-does-tf-app-run-work
# 2) how flags works
# https://stackoverflow.com/questions/33932901/whats-the-purpose-of-tf-app-flags-in-tensorflow
# https://medium.com/@zzzuzum/command-line-flags-python-and-tensorflow-85ab217dbd5

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAG
flags.DEFINE_integer('epoch_num', 20, 'number of epoch')

def main():
  print(FLAGS.epoch_num)

if __name__ == "__main__":
  tf.app.run()


