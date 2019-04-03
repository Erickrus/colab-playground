import com.erickrus.nlp.simp_bert.modeling as modeling
import os
import tensorflow as tf

class WordEmbeddings:
  def __init__(self):
    self.modelBasePath = '/content/drive/My Drive/workspace/bert_model'
    self.modelBaseUrl = 'https://storage.googleapis.com/bert_models'
    self.modelPublishDate = '2018_10_18'

  def download_model(self, modelName = 'uncased_L-12_H-768_A-12'):

    os.system('rm -Rf "%s"' % self.modelBasePath)
    os.system('mkdir -p "%s"' % self.modelBasePath)
    os.system('cd "%s" && wget -c "%s/%s/%s.zip"' % (self.modelBasePath, self.modelBaseUrl, self.modelPublishDate, modelName))
    os.system('cd "%s" && unzip %s.zip -d "%s"' % (self.modelBasePath, modelName, self.modelBasePath))
    os.system('tree "%s"' % self.modelBasePath)

  def extract(self, modelName = 'uncased_L-12_H-768_A-12'):
    modelPath = os.path.join(self.modelBasePath, modelName)
    word_embeddings_variable_name = 'bert/embeddings/word_embeddings'
    word_embeddings_model_name = 'word_embeddings'

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
      for v in tf.global_variables():
        if v.name.find(word_embeddings_variable_name) >=0:
          print(v.name, v.shape)
          embedding_table = v
          break
      
      saver = tf.train.Saver(var_list=[embedding_table])
      saver.save(sess, os.path.join(self.modelBasePath, word_embeddings_model_name))
