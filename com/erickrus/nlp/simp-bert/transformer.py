import tensorflow as tf
import numpy as np
import math


class Transformer:
  def __init__(self):
    self.keepProb = 0.9

  def attention(self, query, key, value, mask=None, dropout=False):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.get_shape().as_list()[-1]
    scores = tf.matmul(query, tf.transpose(key)) / math.sqrt(d_k)
    
    if mask is not None:
      "scores = masked_fill(scores, mask == 0, -1e9)"
      mask = tf.abs(tf.sign(mask))
      kept = tf.ones(mask.shape.as_list()) - mask
      scores = kept * -1e9 + mask * scores
        
    pAttn = tf.nn.softmax(scores, dim = -1)
    if dropout:
      pAttn = tf.nn.dropout(pAttn, keep_prob = self.keepProb)
    return tf.matmul(pAttn, value), pAttn
      

  def multi_headed_attention(self, query, key, value, mask=None, size=512, h=8):
    "Take in model size and number of heads."
    "Implements Figure 2"
    # assert size % h == 0
    # We assume d_v always equals d_k
    d_k = size // h
    
        
    if mask is not None:
      # Same mask applied to all h heads.
      tf.expand_dims(mask, 1)
    nbatches = query.shape[0]

    # 1) Do all the linear projections in batch from d_model => h x d_k 
    #query, key, value = \
    #    [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
    #     for l, x in zip(self.linears, (query, key, value))]     
    
    make_dense = lambda v: \
      tf.transpose(
        tf.reshape(
          tf.layers.dense(v, size),
          [nbatches, -1, h, d_k]
        ),
      perm=[0,2,1,3])
    query, key, value = [make_dense(v) for v in [query, key, value]]
    print(query.shape)
    print(key.shape)
    print(value.shape)
    
    # 2) Apply attention on all the projected vectors in batch. 
    x, attn = self.attention(
                query, key, value, 
                mask=mask, 
                dropout=True
              )

    # 3) "Concat" using a view and apply a final linear. 
    #x = x.transpose(1, 2).contiguous() \
    #     .view(nbatches, -1, self.h * self.d_k)
    x = tf.reshape(
          tf.transpose(x, perm=[0,2,1]), 
          shape=[nbatches, -1, h * d_k]
        )
    return tf.layers.dense(x, size)

  def positionwise_feed_forward(self, x, size= 512):
    "Implements FFN equation."
    d_ff = size*4
    w_2 = tf.layers.dense(
      tf.nn.dropout(
        tf.nn.relu(
          tf.layers.dense(x, d_ff)
        ), keep_prob = self.keepProb
      ), size
    )
    return w_2
  
  def norm(self, x, size=512, eps=1e-6):
    "Construct a layernorm module (See citation for details)."
    a_2, b_2 = tf.Variable(tf.ones(size)), tf.Variable(tf.zeros(size))
    mean, var = tf.nn.moments(x, axes=[1], keep_dims=True)
    varToStd = size / (size - 1 )
    std = tf.sqrt( var * varToStd )
    return a_2 * (x - mean) / (std + eps) + b_2
  
  def sublayer(self, x, sublayer):
    "Apply residual connection to any sublayer with the same size."
    return x + tf.nn.ropout(sublayer(self.norm(x)), keep_prob = self.keepProb)


