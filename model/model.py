import numpy as np
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import dtypes
from common.util import ACTIVATION_MAP
import logging
from tensorflow.contrib.slim.nets import inception
from typing import List, Dict, Set
from common.util import Namespace
from common.losses import get_classifier_loss


tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)


def get_image_size(data_dir: str):
    """ Generates image size based on the dataset directory name

    :param data_dir: path to the data
    :return: image size
    """

    return 299


class ScaledVarianceRandomNormal(init_ops.Initializer):
    """Initializer that generates tensors with a normal distribution scaled as per https://arxiv.org/pdf/1502.01852.pdf.
    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      dtype: The data type. Only floating point types are supported.
    """

    def __init__(self, mean=0.0, factor=1.0, seed=None, dtype=dtypes.float32):
        self.mean = mean
        self.factor = factor
        self.seed = seed
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        if shape:
            n = float(shape[-1])
        else:
            n = 1.0
        for dim in shape[:-2]:
            n *= float(dim)

        self.stddev = np.sqrt(self.factor * 2.0 / n)
        return random_ops.random_normal(shape, self.mean, self.stddev,
                                        dtype, seed=self.seed)
    
class Model:

    def __init__(self, flags: Namespace, embedding_initializer=None, is_training=False):
        self.flags = flags
        self.is_training = is_training
        self.embedding_initializer = embedding_initializer

    def _get_fc_scope(self):
        scope = slim.arg_scope(
            [slim.fully_connected],
            activation_fn=ACTIVATION_MAP[self.flags.activation],
            normalizer_fn=None,
            trainable=True,
            weights_regularizer=None,
        )
        return scope

    def _get_normalizer_params(self):
        normalizer_params = {
            'epsilon': 1e-6,
            'decay': .95,
            'center': True,
            'scale': True,
            'trainable': True,
            'is_training': self.is_training,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'param_regularizers': {'beta': tf.contrib.layers.l2_regularizer(scale=self.flags.weight_decay),
                                   'gamma': tf.contrib.layers.l2_regularizer(scale=self.flags.weight_decay)},
        }
        return normalizer_params

    def _get_scope(self):
        """
        Get slim scope parameters for the convolutional and dropout layers

        :return: convolutional and dropout scopes
        """
        conv2d_arg_scope = slim.arg_scope(
            [slim.conv2d],
            activation_fn=ACTIVATION_MAP[self.flags.activation],
            normalizer_fn=tf.contrib.layers.batch_norm,
            normalizer_params=self._get_normalizer_params(),
            trainable=True,
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.flags.weight_decay),
            weights_initializer=ScaledVarianceRandomNormal(factor=self.flags.weights_initializer_factor),
            biases_initializer=tf.constant_initializer(0.0)
        )
        dropout_arg_scope = slim.arg_scope(
            [slim.dropout],
            keep_prob=self.flags.dropout,
            is_training=self.is_training)
        return conv2d_arg_scope, dropout_arg_scope

    def get_inception_v3(self, images, reuse=None, scope=None):
        arg_scope = inception.inception_v3_arg_scope()
        with slim.arg_scope(arg_scope):
            with tf.variable_scope(scope or 'image_feature_extractor', reuse=reuse):
                images = tf.image.resize_bilinear(images, size=[299, 299], align_corners=False)
                scaled_input_tensor = tf.scalar_mul((1.0 / 255), images)
                scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
                scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

                logits, end_points = inception.inception_v3(scaled_input_tensor,
                                                            is_training=False, num_classes=1001,
                                                            reuse=reuse)
                h = end_points['PreLogits']
                h = slim.flatten(h)
        return h

    def get_projection(self, h, scope="projection"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with self._get_fc_scope():
                if self.flags.embedding_size > 0:
                    h = slim.fully_connected(h, num_outputs=self.flags.embedding_size,
                                             activation_fn=None,
                                             scope='feature_space_layer')
        return h


    def get_image_feature_extractor(self, scope='image_feature_extractor', reuse=None):
        """
            Image feature extractor selector
        :param scope:
        :param reuse:
        :return:
        """
        h = tf.reduce_mean(self.images, axis=1, keepdims=False)

        conv2d_arg_scope, dropout_arg_scope = self._get_scope()
        with conv2d_arg_scope, dropout_arg_scope:
            if self.flags.dropout:
                h = slim.dropout(h, scope='image_dropout', keep_prob=1.0 - self.flags.dropout)

        h = self.get_projection(h, scope="image_projection")
        return h


    def get_cnn_bi_lstm(self, text, text_length, scope='text_feature_extractor', reuse=None):
        """

        :param scope:
        :param reuse:
        :return: the text embedding, BC
        """
        activation_fn = ACTIVATION_MAP[self.flags.activation]
        with tf.variable_scope(scope, reuse=reuse):
            if self.embedding_initializer is not None:
                word_embed_dim = None
                vocab_size = None
            else:
                word_embed_dim = self.flags.word_embed_dim
                vocab_size = self.flags.vocab_size
            h = tf.contrib.layers.embed_sequence(text,
                                                 vocab_size=vocab_size,
                                                 initializer=self.embedding_initializer,
                                                 embed_dim=word_embed_dim,
                                                 trainable=False,
                                                 reuse=tf.AUTO_REUSE,
                                                 scope='TextEmbedding')

            h = tf.expand_dims(h, 1)
            conv2d_arg_scope, dropout_arg_scope = self._get_scope()
            with conv2d_arg_scope, dropout_arg_scope:
                for i in range(self.flags.num_text_cnn_blocks):
                    shortcut = slim.conv2d(h, num_outputs=math.pow(2, i)*self.flags.num_text_cnn_filt, kernel_size=1, stride=1,
                                           activation_fn=None, scope='shortcut' + str(i), padding='SAME')
                    for j in range(self.flags.num_text_cnn_units):
                        h = slim.conv2d(h, num_outputs=math.pow(2, i)*self.flags.num_text_cnn_filt, kernel_size=[1, 3], stride=1,
                                        scope='text_conv' + str(i) + "_" + str(j), padding='SAME', activation_fn=None)
                        if j < (self.flags.num_text_cnn_units - 1):
                            h = activation_fn(h, name='activation_' + str(i) + '_' + str(j))
                    h = h + shortcut
                    h = activation_fn(h, name='activation_' + str(i) + '_' + str(j))

                    h = slim.max_pool2d(h, kernel_size=[1, 2], stride=[1, 2], padding='SAME', scope='text_max_pool' + str(i))
                    text_length = tf.cast(tf.ceil(tf.div(tf.cast(text_length, tf.float32), 2.0)), text_length.dtype)
            h = tf.squeeze(h, [1])

            cells_fw = [tf.nn.rnn_cell.LSTMCell(size) for size in [self.flags.rnn_size]]
            cells_bw = [tf.nn.rnn_cell.LSTMCell(size) for size in [self.flags.rnn_size]]
            h, *_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw, cells_bw=cells_bw,
                                                                   inputs=h, dtype=tf.float32,
                                                                   sequence_length=text_length)

            mask = tf.expand_dims(tf.sequence_mask(text_length,
                                                   maxlen=h.get_shape().as_list()[1], dtype=tf.float32), axis=-1)
            h = tf.reduce_sum(tf.multiply(h, mask), axis=[1]) / tf.reduce_sum(mask, axis=[1])
        return h

    def get_text_feature_extractor(self, scope='text_feature_extractor', reuse=None):
        """
            Text extractor selector
        :param scope:
        :param reuse:
        :return:
        """
        original_shape = self.text.get_shape().as_list()
        if len(original_shape) == 3:
            text = tf.reshape(self.text, shape=([-1]+original_shape[2:]))
            text_length = tf.reshape(self.text_length, shape=[-1])
        else:
            text = self.text
            text_length = self.text_length

        h = self.get_cnn_bi_lstm(text=text, text_length=text_length, reuse=reuse, scope=scope)

        conv2d_arg_scope, dropout_arg_scope = self._get_scope()
        with conv2d_arg_scope, dropout_arg_scope:
            if self.flags.dropout:
                h = slim.dropout(h, scope='text_dropout', keep_prob=1.0 - self.flags.dropout)

        h = self.get_projection(h, scope="text_projection")

        if len(original_shape) == 3:
            h = tf.reshape(h, shape=([-1] + [original_shape[1], h.get_shape().as_list()[-1]]))
            h = tf.reduce_mean(h, axis=1, keepdims=False)
        return h

    def get_distance_head(self, embedding_mod1, embedding_mod2, scope='distance_head'):
        """
            Implements the a distance head, measuring distance between elements in embedding_mod1 and embedding_mod2.
            Input dimensions are B1C and B2C, output dimentions are B1B2. The distance between diagonal elements is supposed to be small.
            The distance between off-diagonal elements is supposed to be large. Output can be considered to be classification logits.
        :param embedding_mod1: embedding of modality one, B1C
        :param embedding_mod2: embeddings of modality two, B2C
        :param scope:
        :return:
        """
        with tf.variable_scope(scope):
            # i is the number of elements in the embedding_mod1 batch
            # j is the number of elements in the embedding_mod2 batch
            j = embedding_mod2.get_shape()[0]
            i = embedding_mod1.get_shape()[0]
            # tile to be able to produce weight matrix alpha in (i,j) space
            embedding_mod1 = tf.expand_dims(embedding_mod1, axis=1)
            embedding_mod2 = tf.expand_dims(embedding_mod2, axis=0)
            # features_generic changes over i and is constant over j
            # task_encoding changes over j and is constant over i
            embedding_mod2_tile = tf.tile(embedding_mod2, (i, 1, 1))
            embedding_mod1_tile = tf.tile(embedding_mod1, (1, j, 1))
            # Compute distance
            euclidian = -tf.norm(embedding_mod2_tile - embedding_mod1_tile, name='neg_euclidian_distance', axis=-1)
            return euclidian

    def get_metric(self, image_embeddings, text_embeddings):
        with tf.variable_scope('Metric'):
            logits = self.get_distance_head(embedding_mod1=image_embeddings, embedding_mod2=text_embeddings,
                                            scope='distance_head')
        return logits

    def get_embeddings(self, reuse=False):
        with tf.variable_scope('Model', reuse=reuse):
            self.image_embeddings = self.get_image_feature_extractor(scope='image_feature_extractor', reuse=reuse)
            self.text_embeddings = self.get_text_feature_extractor(scope='text_feature_extractor', reuse=reuse)

    def get_inference_graph(self, reuse=False):
        """
            Creates text embedding, image embedding and links them using a distance metric.
            Ouputs logits that can be used for training and inference, as well as text and image embeddings.
        :param reuse:
        :return:
        """
        self.get_embeddings(reuse=reuse)
        self.logits = self.get_metric(image_embeddings=self.image_embeddings,
                                      text_embeddings=self.text_embeddings)

    def get_input_placeholders(self, batch_size_image: int = 32, batch_size_text: int = None,
                               num_images: int = 10, num_texts: int = 10,
                               max_text_len: int = 30, scope: str = "input"):
        """
        :param num_images:
        :param num_texts:
        :param max_text_len:
        :param scope:
        :return:
        :param batch_size_image: the number of image instances (image instance features) in the batch
        :param batch_size_text: the number of text instances (text instance features or prototypes) in the batch
        :return: placeholders for images, text and class labels
        """
        if batch_size_text is None:
            batch_size_text = batch_size_image

        with tf.variable_scope(scope):
            self.images = self.get_images_placeholder(batch_size=batch_size_image, num_images=num_images)
            self.text = tf.placeholder(shape=(batch_size_text, num_texts, max_text_len), name='text', dtype=tf.int32)
            self.text_length = tf.placeholder(shape=(batch_size_text, num_texts), name='text_len', dtype=tf.int32)
            self.labels_txt2img = tf.placeholder(tf.int64, shape=(batch_size_text,), name='match_labels_txt2img')
            self.labels_img2txt = tf.placeholder(tf.int64, shape=(batch_size_image,), name='match_labels_img2txt')
            self.labels_class = tf.placeholder(tf.int64, shape=(batch_size_image,), name='class_labels')

    def get_images_placeholder(self, batch_size: int, num_images: int):
        images_placeholder = tf.placeholder(tf.float32, name='images', shape=(batch_size, num_images, 2048))
        return images_placeholder


    def get_lr(self, global_step=None):
        """
        Creates a learning rate schedule
        :param global_step: external global step variable, if None new one is created here
        :return:
        """
        if global_step is None:
            global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)

        if self.flags.lr_anneal == 'exp':
            lr_decay_step = self.flags.number_of_steps // self.flags.n_lr_decay
            learning_rate = tf.train.exponential_decay(self.flags.init_learning_rate, global_step, lr_decay_step,
                                                       1.0 / self.flags.lr_decay_rate, staircase=True)
        else:
            raise Exception('Learning rate schedule not implemented')

        tf.summary.scalar('learning_rate', learning_rate)
        self.learning_rate = learning_rate


    def get_main_train_op(self, loss: tf.Tensor, global_step: tf.Variable):
        """
        Creates a train operation to minimize loss
        :param loss: loss to be minimized
        :param global_step: global step to be incremented whilst invoking train opeation created
        :param flags: overall architecture parameters
        :return:
        """

        # Learning rate
        self.get_lr(global_step=global_step)
        # Optimizer
        if self.flags.optimizer == 'sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
        elif self.flags.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            raise Exception('Optimizer not implemented')
        # get variables to train
        variables_to_train = tf.trainable_variables(scope='(?!.*image_feature_extractor).*')
        # Train operation
        return slim.learning.create_train_op(total_loss=loss, optimizer=optimizer, global_step=global_step,
                                             clip_gradient_norm=self.flags.clip_gradient_norm,
                                             variables_to_train=variables_to_train)

