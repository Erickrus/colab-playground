import com.erickrus.nlp.simp_bert.modeling as modeling
import os
import tensorflow as tf
import numpy as np

class WordEmbeddings:
  def __init__(self):
    self.modelBasePath = '/content/drive/My Drive/workspace/bert_model'
    self.modelBaseUrl = 'https://storage.googleapis.com/bert_models'
    self.modelPublishDate = '2018_10_18'
    self.word_embeddings_model_name = 'word_embeddings.npy'
    self.word_embeddings_variable_name = 'bert/embeddings/word_embeddings'

  def download_model(self, modelName = 'uncased_L-12_H-768_A-12'):
    print('download_model()')
    os.system('rm -Rf "%s"' % self.modelBasePath)
    os.system('mkdir -p "%s"' % self.modelBasePath)
    os.system('cd "%s" && wget -c "%s/%s/%s.zip"' % (self.modelBasePath, self.modelBaseUrl, self.modelPublishDate, modelName))
    os.system('cd "%s" && unzip %s.zip -d "%s"' % (self.modelBasePath, modelName, self.modelBasePath))
    os.system('tree "%s"' % self.modelBasePath)

  def find_tensor_by_name(self, tensorName):
    for v in tf.global_variables():
      if v.name.find(tensorName) >=0:
        self.inspect(v)
        return v

  def inspect(self, tensor):
    print(tensor.name, tensor.shape, tensor.dtype)

  def extract_model(self, modelName = 'uncased_L-12_H-768_A-12'):
    print('extract_model()')
    modelPath = os.path.join(self.modelBasePath, modelName)
    # bert/embeddings/word_embeddings:0 (30522, 768)
    
    config = modeling.BertConfig.from_json_file(os.path.join(modelPath, "bert_config.json"))

    # input_ids: int32 Tensor of shape [batch_size, seq_length].
    seq_length = 384
    batch_size = 4
    input_ids = tf.reshape(tf.zeros(batch_size * seq_length, dtype=tf.int32), [batch_size, seq_length])
    model = modeling.BertModel(
      config,
      is_training=False,
      input_ids=input_ids
    )

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)
      tvars = tf.trainable_variables()
      init_checkpoint = os.path.join(modelPath, "bert_model.ckpt")
      
      (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
      embedding_table = self.find_tensor_by_name(self.word_embeddings_variable_name)
      
      npyWordEmbedding = sess.run(embedding_table)
      npyWordEmbeddingFilename = os.path.join(self.modelBasePath, self.word_embeddings_model_name)
      np.save(open(npyWordEmbeddingFilename, 'wb'), npyWordEmbedding)

  def load_word_embeddings(self, variableName):
    # https://www.aiworkbox.com/lessons/initialize-a-tensorflow-variable-with-numpy-values
    # using initializer to load from numpy object directly

    # https://www.tensorflow.org/api_docs/python/tf/get_variable
    # how to use variable_scope and reuse
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
      return tf.get_variable(
        variableName, 
        initializer=np.load(
          open(
            os.path.join(
              self.modelBasePath, 
              self.word_embeddings_model_name
            ), 
          'rb')
        )
      )

"""

Another solution is:
with tf.variable_scope("encoder")
   ...
vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
saver = tf.train.Saver(vars)
saver.save(sess=sess, save_path="...")

"""