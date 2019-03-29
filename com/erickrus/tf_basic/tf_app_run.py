# This file is to demostrate
# 1) how tf.app.run() works
# https://stackoverflow.com/questions/33703624/how-does-tf-app-run-work
# 2) how flags works
# https://stackoverflow.com/questions/33932901/whats-the-purpose-of-tf-app-flags-in-tensorflow
# https://medium.com/@zzzuzum/command-line-flags-python-and-tensorflow-85ab217dbd5

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epoch_num', 20, 'number of epoch')

# notice that main will take some parameters,
# so just give the anonymous _
def main(_):
  print(FLAGS.epoch_num)

if __name__ == "__main__":
  tf.app.run()

# use following command to execute it in the colab
# ! cd "/content/drive/My Drive/Colab Notebooks/colab-playground" && python3 -u com/erickrus/tf_basic/tf_app_run.py
# ! cd "/content/drive/My Drive/Colab Notebooks/colab-playground" && python3 -u com/erickrus/tf_basic/tf_app_run.py --epoch_num 99

