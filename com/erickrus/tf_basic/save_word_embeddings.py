"""
# download bert model
!rm -Rf  /content/drive/My\ Drive/workspace/bert_model
!mkdir -p /content/drive/My\ Drive/workspace/bert_model
!cd /content/drive/My\ Drive/workspace/bert_model && wget -c https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!cd /content/drive/My\ Drive/workspace/bert_model && unzip uncased_L-12_H-768_A-12.zip -d /content/drive/My\ Drive/workspace/bert_model
!tree /content/drive/My\ Drive/workspace/bert_model
"""



import com.erickrus.nlp.simp_bert.modeling as modeling
import os
import tensorflow as tf
modelPath = "/content/drive/My Drive/workspace/bert_model/uncased_L-12_H-768_A-12/"
configFilename = os.path.join(modelPath, "bert_config.json")
config = modeling.BertConfig.from_json_file(configFilename)

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
  #print(1)
  (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
  tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
  for v in tf.global_variables():
    if v.name.find("bert/embeddings/word_embeddings") >=0:
      print(v.name, v.shape)
      embedding_table = v
      break
  
  saver = tf.train.Saver(var_list=[embedding_table])
  saver.save(sess, '/content/drive/My Drive/workspace/bert_model/word_embeddings')       # saving the variable 
