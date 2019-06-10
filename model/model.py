import os
import numpy as np
import math
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import dtypes
from common.util import ACTIVATION_MAP
import pathlib
import logging
from common.util import summary_writer
from common.gen_experiments import load_and_save_params
import time
from tqdm import tqdm, trange
from tensorflow.contrib.slim.nets import inception
from typing import List, Dict, Set
from common.util import Namespace
from datasets import Dataset
from datasets.dataset_list import get_dataset_splits
from common.metrics import ap_at_k_prototypes, top1_gzsl
from common.pretrained_models import IMAGE_MODEL_CHECKPOINTS, get_image_fe_restorer
from common.losses import get_classifier_loss


tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)


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
    
    
def _get_fc_scope(is_training, flags):
    scope = slim.arg_scope(
        [slim.fully_connected],
        activation_fn=ACTIVATION_MAP[flags.activation],
        normalizer_fn=None,
        trainable=True,
        weights_regularizer=None,
    )
    return scope


def _get_normalizer_params(is_training, flags):
    normalizer_params = {
        'epsilon': 1e-6,
        'decay': .95,
        'center': True,
        'scale': True,
        'trainable': True,
        'is_training': is_training,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,  
        'param_regularizers': {'beta': tf.contrib.layers.l2_regularizer(scale=flags.weight_decay),
                               'gamma': tf.contrib.layers.l2_regularizer(scale=flags.weight_decay)},
    }
    return normalizer_params


def _get_scope(is_training, flags):
    """
    Get slim scope parameters for the convolutional and dropout layers

    :param is_training: whether the network is in training mode
    :param flags: overall settings of the model
    :return: convolutional and dropout scopes
    """
    normalizer_params = _get_normalizer_params(is_training, flags)
    conv2d_arg_scope = slim.arg_scope(
        [slim.conv2d],
        activation_fn=ACTIVATION_MAP[flags.activation],
        normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params=normalizer_params,
        trainable=True,
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=flags.weight_decay),
        weights_initializer=ScaledVarianceRandomNormal(factor=flags.weights_initializer_factor),
        biases_initializer=tf.constant_initializer(0.0)
    )
    dropout_arg_scope = slim.arg_scope(
        [slim.dropout],
        keep_prob=flags.dropout,
        is_training=is_training)
    return conv2d_arg_scope, dropout_arg_scope


def get_inception_v3(images, flags, is_training=False, reuse=None, scope=None):
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

                
def get_projection(h, flags, is_training, scope="projection"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        with _get_fc_scope(is_training, flags):
            if flags.embedding_size > 0:
                h = slim.fully_connected(h, num_outputs=flags.embedding_size, 
                                         activation_fn=None,
                                         scope='feature_space_layer')
    return h


def get_image_feature_extractor(images: tf.Tensor, flags, is_training=False, scope='image_feature_extractor', reuse=None):
    """
        Image feature extractor selector
    :param images: tensor of input images in the format BHWC
    :param flags: overall architecture settings
    :param is_training:
    :param scope:
    :param reuse:
    :return:
    """
    h = tf.reduce_mean(images, axis=1, keepdims=False)
        
    conv2d_arg_scope, dropout_arg_scope = _get_scope(is_training, flags)
    with conv2d_arg_scope, dropout_arg_scope:
        if flags.dropout:
            h = slim.dropout(h, scope='image_dropout', keep_prob=1.0 - flags.dropout)
            
    h = get_projection(h, flags=flags, is_training=is_training, scope="image_projection")            
    return h


def get_cnn_bi_lstm(text, text_length, flags, embedding_initializer=None,
                       is_training=False, scope='text_feature_extractor', reuse=None):
    """

    :param text: input text sequence, BTC
    :param text_length:  lengths of sequences in the batch, B
    :param flags:  general settings of the overall architecture
    :param is_training:
    :param scope:
    :param reuse:
    :return: the text embedding, BC
    """
    activation_fn = ACTIVATION_MAP[flags.activation]
    with tf.variable_scope(scope, reuse=reuse):
        if embedding_initializer is not None:
            word_embed_dim = None
            vocab_size = None
        else:
            word_embed_dim = flags.word_embed_dim
            vocab_size = flags.vocab_size
        h = tf.contrib.layers.embed_sequence(text,
                                             vocab_size=vocab_size,
                                             initializer=embedding_initializer,
                                             embed_dim=word_embed_dim,
                                             trainable=False,
                                             reuse=tf.AUTO_REUSE,
                                             scope='TextEmbedding')
        
        h = tf.expand_dims(h, 1)
        conv2d_arg_scope, dropout_arg_scope = _get_scope(is_training, flags)
        normalizer_params = _get_normalizer_params(is_training, flags)
        with conv2d_arg_scope, dropout_arg_scope:
            for i in range(flags.num_text_cnn_blocks):
                shortcut = slim.conv2d(h, num_outputs=math.pow(2, i)*flags.num_text_cnn_filt, kernel_size=1, stride=1,
                                       activation_fn=None, scope='shortcut' + str(i), padding='SAME')
                for j in range(flags.num_text_cnn_units):
                    h = slim.conv2d(h, num_outputs=math.pow(2, i)*flags.num_text_cnn_filt, kernel_size=[1, 3], stride=1,
                                    scope='text_conv' + str(i) + "_" + str(j), padding='SAME', activation_fn=None)
                    if j < (flags.num_text_cnn_units - 1):
                        h = activation_fn(h, name='activation_' + str(i) + '_' + str(j))
                h = h + shortcut
                h = activation_fn(h, name='activation_' + str(i) + '_' + str(j))

                h = slim.max_pool2d(h, kernel_size=[1, 2], stride=[1, 2], padding='SAME', scope='text_max_pool' + str(i))
                text_length = tf.cast(tf.ceil(tf.div(tf.cast(text_length, tf.float32), 2.0)), text_length.dtype)
        h = tf.squeeze(h, [1])        

        cells_fw = [tf.nn.rnn_cell.LSTMCell(size) for size in [flags.rnn_size]]
        cells_bw = [tf.nn.rnn_cell.LSTMCell(size) for size in [flags.rnn_size]]
        h, *_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw, cells_bw=cells_bw,
                                                               inputs=h, dtype=tf.float32,
                                                               sequence_length=text_length)
        
        mask = tf.expand_dims(tf.sequence_mask(text_length, 
                                               maxlen=h.get_shape().as_list()[1], dtype=tf.float32), axis=-1)
        h = tf.reduce_sum(tf.multiply(h, mask), axis=[1]) / tf.reduce_sum(mask, axis=[1])
        
    return h


def get_text_feature_extractor(text, text_length, flags, embedding_initializer=None,
                               is_training=False, scope='text_feature_extractor', reuse=None):
    """
        Text extractor selector
    :param text: tensor of input texts tokenized as integers in the format BL
    :param text_length: tensor of sequence lengths
    :param flags: overall architecture settings
    :param embedding_size: the length of embedding vector
    :param is_training:
    :param scope:
    :param reuse:
    :return:
    """
    original_shape = text.get_shape().as_list()
    if len(original_shape) == 3:
        text = tf.reshape(text, shape=([-1]+original_shape[2:]))
        text_length = tf.reshape(text_length, shape=[-1])
        
    h = get_cnn_bi_lstm(text, text_length, flags=flags,
                        embedding_initializer=embedding_initializer, is_training=is_training,
                        reuse=reuse, scope=scope)
            
    conv2d_arg_scope, dropout_arg_scope = _get_scope(is_training, flags)
    with conv2d_arg_scope, dropout_arg_scope:
        if flags.dropout:
            h = slim.dropout(h, scope='text_dropout', keep_prob=1.0 - flags.dropout)

    h = get_projection(h, flags=flags, is_training=is_training, scope="text_projection")
    
    if len(original_shape) == 3:
        h = tf.reshape(h, shape=([-1] + [original_shape[1], h.get_shape().as_list()[-1]]))
        h = tf.reduce_mean(h, axis=1, keepdims=False)
    return h


def get_distance_head(embedding_mod1, embedding_mod2, flags, is_training, scope='distance_head'):
    """
        Implements the a distance head, measuring distance between elements in embedding_mod1 and embedding_mod2.
        Input dimensions are B1C and B2C, output dimentions are B1B2. The distance between diagonal elements is supposed to be small.
        The distance between off-diagonal elements is supposed to be large. Output can be considered to be classification logits.
    :param embedding_mod1: embedding of modality one, B1C
    :param embedding_mod2: embeddings of modality two, B2C
    :param flags: general architecture parameters
    :param is_training:
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
    

def get_metric(image_embeddings, text_embeddings, flags, is_training, reuse=False):
    with tf.variable_scope('Metric', reuse=reuse):
        logits = get_distance_head(embedding_mod1=image_embeddings, embedding_mod2=text_embeddings,
                                       flags=flags, is_training=is_training, scope='distance_head')
    return logits


def get_inference_graph(images, text, text_length, flags, is_training, embedding_initializer=None, reuse=False):
    """
        Creates text embedding, image embedding and links them using a distance metric.
        Ouputs logits that can be used for training and inference, as well as text and image embeddings.
    :param images:
    :param text:
    :param text_length:
    :param flags:
    :param is_training:
    :param reuse:
    :return:
    """
    image_embeddings, text_embeddings = get_embeddings(images, text, text_length,
                                                       flags=flags, is_training=is_training,
                                                       embedding_initializer=embedding_initializer, reuse=reuse)
    logits = get_metric(image_embeddings, text_embeddings,
                        flags=flags, is_training=is_training, reuse=reuse)

    return logits, image_embeddings, text_embeddings


def get_embeddings(images, text, text_length, flags, is_training, embedding_initializer=None, reuse=False):
    with tf.variable_scope('Model', reuse=reuse):
        image_embeddings = get_image_feature_extractor(images, flags, is_training=is_training,
                                                       scope='image_feature_extractor', reuse=reuse)
        text_embeddings = get_text_feature_extractor(text, text_length, flags,
                                                     embedding_initializer=embedding_initializer,
                                                     is_training=is_training, scope='text_feature_extractor',
                                                     reuse=reuse)
    return image_embeddings, text_embeddings


def get_input_placeholders(batch_size_image: int = 32, batch_size_text: int = None,
                           image_size: int = 299, num_images: int = 10, num_texts: int = 10,
                           max_text_len: int = 30, flags: Namespace = None, scope: str = "input"):
    """
    :param image_size:
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
        images_placeholder = get_images_placeholder(batch_size=batch_size_image, num_images=num_images,
                                                    image_size=image_size, flags=flags)
        text_placeholder = tf.placeholder(shape=(batch_size_text, num_texts, max_text_len), name='text', dtype=tf.int32)
        text_length_placeholder = tf.placeholder(shape=(batch_size_text, num_texts), name='text_len', dtype=tf.int32)
        labels_txt2img = tf.placeholder(tf.int64, shape=(batch_size_text,), name='match_labels_txt2img')
        labels_img2txt = tf.placeholder(tf.int64, shape=(batch_size_image,), name='match_labels_img2txt')
        labels_class = tf.placeholder(tf.int64, shape=(batch_size_image,), name='class_labels')
    return images_placeholder, text_placeholder, text_length_placeholder, labels_txt2img, labels_img2txt, labels_class


def get_image_size(data_dir: str):
    """ Generates image size based on the dataset directory name

    :param data_dir: path to the data
    :return: image size
    """

    return 299


def get_images_placeholder(batch_size: int, num_images: int, image_size: int, flags: Namespace):
    num_features_dict = {'inception_v3': 2048, 'resnet101': 2048}
    images_placeholder = tf.placeholder(tf.float32, name='images', 
                                        shape=(batch_size, num_images,
                                               num_features_dict[flags.image_feature_extractor]))
    return images_placeholder


def get_lr(global_step=None, flags=None):
    """
    Creates a learning rate schedule
    :param global_step: external global step variable, if None new one is created here
    :param flags:
    :return:
    """
    if global_step is None:
        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)

    if flags.lr_anneal == 'exp':
        lr_decay_step = flags.number_of_steps // flags.n_lr_decay
        learning_rate = tf.train.exponential_decay(flags.init_learning_rate, global_step, lr_decay_step,
                                                   1.0 / flags.lr_decay_rate, staircase=True)
    else:
        raise Exception('Learning rate schedule not implemented')

    tf.summary.scalar('learning_rate', learning_rate)
    return learning_rate


def get_main_train_op(loss: tf.Tensor, global_step: tf.Variable, flags: Namespace):
    """
    Creates a train operation to minimize loss
    :param loss: loss to be minimized
    :param global_step: global step to be incremented whilst invoking train opeation created
    :param flags: overall architecture parameters
    :return:
    """

    # Learning rate
    learning_rate = get_lr(global_step=global_step, flags=flags)
    # Optimizer
    if flags.optimizer == 'sgd':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif flags.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        raise Exception('Optimizer not implemented')
    # get variables to train
    variables_to_train = tf.trainable_variables(scope='(?!.*image_feature_extractor).*')
    # Train operation
    return slim.learning.create_train_op(total_loss=loss, optimizer=optimizer, global_step=global_step,
                                         clip_gradient_norm=flags.clip_gradient_norm,
                                         variables_to_train=variables_to_train)

