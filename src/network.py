import tensorflow as tf
from .modules.embedding import Embedding
from .modules.encoder import Encoder
from .modules.alignment import registry as alignment
from .modules.fusion import registry as fusion
from .modules.connection import registry as connection
from .modules.pooling import pooling
from .modules.prediction import registry as prediction
from .modules.__init__ import *
from tensorflow.contrib.framework import arg_scope

def layer_norm (input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


class Network:
    def __init__ (self, args):
        self.embedding = Embedding(args)
        self.blocks = [{
            'encoder': Encoder(args),
            'alignment': alignment[args.alignment](args),
            'fusion': fusion[args.fusion](args),
        } for _ in range(args.blocks)]
        self.connection = connection[args.connection]
        self.pooling = pooling
        self.prediction = prediction[args.prediction](args)
        self.max_len = args.max_len

    def __call__ (self, a, b, mask_a, mask_b, dropout_keep_prob):

        a = self.embedding(a, dropout_keep_prob)
        b = self.embedding(b, dropout_keep_prob)

        res_a, res_b = a, b

        for i, block in enumerate(self.blocks):
            with tf.variable_scope('block-{}'.format(i), reuse=tf.AUTO_REUSE):
                if i > 0:
                    a = self.connection(a, res_a, i)
                    b = self.connection(b, res_b, i)
                    res_a, res_b = a, b

                a_enc = block['encoder'](a, mask_a, dropout_keep_prob)
                b_enc = block['encoder'](b, mask_b, dropout_keep_prob)
                a = tf.concat([a, a_enc], axis=-1)
                b = tf.concat([b, b_enc], axis=-1)


                align_a, align_b = block['alignment'](a, b, mask_a, mask_b, dropout_keep_prob)
                a = block['fusion'](a, align_a, dropout_keep_prob)
                b = block['fusion'](b, align_b, dropout_keep_prob)

        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)

        out = self.prediction(a, b, dropout_keep_prob)
        adv_out1 = self.get_adv_out(a, reuse=False)
        adv_out2 = self.get_adv_out(b, reuse=True)


        return out, adv_out1, adv_out2


    def get_adv_out (self, X, reuse):
        with tf.variable_scope("adv", reuse=reuse):
            layers1 = dense(X, 200, None, name='dense1')
            layers1 = dense (layers1, 1, None, name='dense2')
            return layers1

