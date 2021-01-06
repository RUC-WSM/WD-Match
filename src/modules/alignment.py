


import math
import tensorflow as tf
from functools import partial
from src.utils.registry import register
from . import dense

registry = {}
register = partial(register, registry=registry)


@register('identity')
class Alignment:
    def __init__(self, args):
        self.args = args

    def _attention(self, a, b, t, _):
        return tf.matmul(a, b, transpose_b=True) * t

    def __call__(self, a, b, mask_a, mask_b, dropout_keep_prob, name='alignment'):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            feature_a = self.decoder(a,b,b,mask_a,mask_b)
            feature_b = self.decoder(b,a,a, mask_b, mask_a)

            return feature_a, feature_b



    def decoder(self,Q,K,V,mask_a,mask_b,name="default"):
        decoder_numer  = self.args.decoder_num
        Q_out = Q
        for i in range(decoder_numer):
            with tf.variable_scope(name+str(i)):
                temperature = tf.get_variable('temperature', shape=(), dtype=tf.float32, trainable=True,
                                              initializer=tf.constant_initializer(math.sqrt(1 / self.args.hidden_size)))
                attention = tf.matmul(Q_out, K, transpose_b=True) * temperature
                attention_mask = tf.matmul(mask_a, mask_b, transpose_b=True)
                attention = attention_mask * attention + (1 - attention_mask) * tf.float32.min
                attention_Q = tf.nn.softmax(attention, dim=2)
                
                Q_out = tf.matmul(attention_Q, V)

        return Q_out




@register('linear')
class MappedAlignment(Alignment):
    def _attention(self, a, b, t, dropout_keep_prob):
        with tf.variable_scope(f'proj'):
            a = dense(tf.nn.dropout(a, dropout_keep_prob),
                      self.args.hidden_size, activation=tf.nn.relu)
            b = dense(tf.nn.dropout(b, dropout_keep_prob),
                      self.args.hidden_size, activation=tf.nn.relu)
            return super()._attention(a, b, t, dropout_keep_prob)
