#!/usr/bin/env python3

"""Training and evaluation entry point."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from common.pretrained_models import IMAGE_MODEL_CHECKPOINTS
from common.losses import get_rmse_loss, get_mi_loss, get_dist_mtx, get_cross_classifier_loss, get_som_loss, get_classifier_loss


tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'test'])
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the data.')
    parser.add_argument('--train_split', type=str, default='trainval', choices=['train', 'trainval'],
                        help='Split of the data to be used to perform operation.')
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'val'],
                        help='Split of the data to be used to perform operation.')
    parser.add_argument('--dataset', type=str, default='xian2017_cub',
                        choices=['cvpr2016_cub', 'xian2017_cub', 'xian2018_flowers'], help='Dataset to train.')

    # Training parameters
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--number_of_steps', type=int, default=int(100001),
                        help="Number of training steps (number of Epochs in Hugo's paper)")
    parser.add_argument('--number_of_steps_to_early_stop', type=int, default=int(1000000),
                        help="Number of training steps after half way to early stop the training")
    parser.add_argument('--log_dir', type=str, default='', help='Base log dir')
    parser.add_argument('--exp_dir', type=str, default=None, help='experiement directory for Borgy')
    # Batch parameters
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--num_images', type=int, default=1, help='Number of image samples per image/text pair.')
    parser.add_argument('--num_texts', type=int, default=10, help='Number of text samples per image/text pair.')
    parser.add_argument('--init_learning_rate', type=float, default=0.1005, help='Initial learning rate.')
    parser.add_argument('--save_summaries_secs', type=int, default=60, help='Time between saving summaries')
    parser.add_argument('--save_interval_secs', type=int, default=60, help='Time between saving model?')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--augment', type=bool, default=False)
    # Learning rate paramteres
    parser.add_argument('--lr_anneal', type=str, default='exp', choices=['exp'])
    parser.add_argument('--n_lr_decay', type=int, default=3)
    parser.add_argument('--lr_decay_rate', type=float, default=2.0)
    parser.add_argument('--num_steps_decay_pwc', type=int, default=2500,
                        help='Decay learning rate every num_steps_decay_pwc')

    parser.add_argument('--clip_gradient_norm', type=float, default=1.0, help='gradient clip norm.')
    parser.add_argument('--weights_initializer_factor', type=float, default=0.1,
                        help='multiplier in the variance of the initialization noise.')
    # Evaluation parameters
    parser.add_argument('--max_number_of_evaluations', type=float, default=float('inf'))
    parser.add_argument('--eval_interval_secs', type=int, default=120, help='Time between evaluating model?')
    parser.add_argument('--eval_interval_steps', type=int, default=2500,
                        help='Number of train steps between evaluating model in the training loop')
    parser.add_argument('--eval_interval_fine_steps', type=int, default=1000,
                        help='Number of train steps between evaluating model in the training loop in the final phase')
    parser.add_argument('--num_samples_eval', type=int, default=100, help='Number of evaluation samples?')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size?')
    # Test parameters
    parser.add_argument('--pretrained_model_dir', type=str,
                        default='./logs/batch_size-32-lr-0.122-lr_anneal-cos-epochs-100.0-dropout-1.0-optimizer-sgd-weight_decay-0.0005-augment-False-num_filters-64-feature_extractor-simple_res_net-task_encoder-class_mean-attention_num_filters-32/train',
                        help='Path to the pretrained model to run the nearest neigbor baseline test.')
    # Architecture parameters
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--embedding_pooled', type=bool, default=True)
    # Image feature extractor
    parser.add_argument('--image_feature_extractor', type=str, default='resnet101',
                        choices=['simple_res_net', 'inception_v3', 'inception_v2', 'resnet101'], help='Which feature extractor to use')
    parser.add_argument('--image_fe_trainable', type=bool, default=False)
    parser.add_argument('--num_filters', type=int, default=64)
    parser.add_argument('--num_units_in_block', type=int, default=3)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--num_max_pools', type=int, default=3)
    parser.add_argument('--block_size_growth', type=float, default=2.0)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'selu', 'swish-1'])
    parser.add_argument('--train_bn_proba', type=float, default=1.0,
                        help='Probability that batch norm/dropout layers are in train mode during training')
    # Text feature extractor
    parser.add_argument('--word_embed_trainable', type=bool, default=False)
    parser.add_argument('--word_embed_dim', type=int, default=300) # this should be equal to the word2vec dimension
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--text_feature_extractor', type=str, default='cnn_bi_lstm',
                        choices=['simple_bi_lstm', 'cnn_bi_lstm', '2016cnn_bi_lstm'])
    parser.add_argument('--text_maxlen', type=int, default=30, help='Maximal length of the text description in tokens')
    parser.add_argument('--shuffle_text_in_batch', type=bool, default=False)
    parser.add_argument('--rnn_size', type=int, default=512)
    parser.add_argument('--num_text_cnn_filt', type=int, default=256)
    parser.add_argument('--num_text_cnn_units', type=int, default=3)
    parser.add_argument('--num_text_cnn_blocks', type=int, default=2)
    
    
    parser.add_argument('--embedding_size', type=int, default=1024)
    parser.add_argument('--latent_dim', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=0)
    

    parser.add_argument('--metric_multiplier_init', type=float, default=5.0, help='multiplier of cosine metric')
    parser.add_argument('--metric_multiplier_trainable', type=bool, default=False,
                        help='multiplier of cosine metric trainability')
    parser.add_argument('--polynomial_metric_order', type=int, default=1)
    # Cross modal consistency loss
    parser.add_argument('--mi_weight', type=float, default=None,
                        help='The weight of the mutual information term between text and image distances')
    parser.add_argument('--mi_kernel_width', type=float, default=1.0,
                        help='The width of KDE kernel used to estmiate MI')
    parser.add_argument('--mi_train_offset', type=float, default=0.0,
                        help='The proportion of steps to delay the inclusion of MI loss into total loss')
    parser.add_argument('--consistency_loss', type=str, default=None, choices=[None, "NMSE", "MI", "CROSSCLASS", 
                                                                                "SOM", "CLASSIFIER"])
    parser.add_argument('--cross_class_num_clusters', type=int, default=1024)
    parser.add_argument('--cross_class_metric_scale', type=float, default=100.0)
    parser.add_argument('--cross_class_decay', type=float, default=0.9)
    parser.add_argument('--cross_class_sigma_0', type=float, default=1.0)
    parser.add_argument('--num_classes_train', type=int, default=250)
    parser.add_argument('--weight_decay_fc', type=float, default=0.001)
    
    parser.add_argument('--modality_interaction', type=str, default="None", choices=["None", "FILM"])
    parser.add_argument('--film_weight_decay', type=float, default=0.001)
    parser.add_argument('--film_weight_decay_postmult', type=float, default=0.1)
    


    args = parser.parse_args()

    print(args)
    return args


def get_image_size(data_dir: str):
    """ Generates image size based on the dataset directory name

    :param data_dir: path to the data
    :return: image size
    """

    return 299


def get_logdir_name(flags):
    """Generates the name of the log directory from the values of flags
    Parameters
    ----------
        flags: neural net architecture generated by get_arguments()
    Outputs
    -------
        the name of the directory to store the training and evaluation results
    """

    param_list = ['batch_size', str(flags.train_batch_size), 'steps', str(flags.number_of_steps),
                  'lr', str(flags.init_learning_rate), 'opt', flags.optimizer,
                  'weight_decay', str(flags.weight_decay),
                  'nfilt', str(flags.num_filters), 'image_feature_extractor', str(flags.image_feature_extractor),
                  ]

    if flags.log_dir == '':
        logdir = './logs/' + '-'.join(param_list)
    else:
        logdir = os.path.join(flags.log_dir, '-'.join(param_list))

    if flags.exp_dir is not None:
        # Running a Borgy experiment
        logdir = flags.exp_dir

    return logdir
    

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


def _get_film_fc_scope(is_training, flags):
    scope = slim.arg_scope(
        [slim.fully_connected],
        activation_fn=ACTIVATION_MAP[flags.activation],
        normalizer_fn=None,
        trainable=True,
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=flags.film_weight_decay),
        weights_initializer=ScaledVarianceRandomNormal(factor=flags.weights_initializer_factor),
    )
    return scope
    
    
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
        'updates_collections': tf.GraphKeys.UPDATE_OPS,  # [tf.GraphKeys.UPDATE_OPS, None]
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


def get_simple_res_net(images, flags, num_filters, is_training=False, reuse=None, scope=None):
    conv2d_arg_scope, dropout_arg_scope = _get_scope(is_training, flags)
    activation_fn = ACTIVATION_MAP[flags.activation]
    with conv2d_arg_scope, dropout_arg_scope:
        with tf.variable_scope(scope or 'image_feature_extractor', reuse=reuse):
            # h = slim.conv2d(images, num_outputs=num_filters[0], kernel_size=6, stride=1,
            #                 scope='conv_input', padding='SAME')
            # h = slim.max_pool2d(h, kernel_size=2, stride=2, padding='SAME', scope='max_pool_input')
            h = images
            for i in range(len(num_filters)):
                # make shortcut
                shortcut = slim.conv2d(h, num_outputs=num_filters[i], kernel_size=1, stride=1,
                                       activation_fn=None,
                                       scope='shortcut' + str(i), padding='SAME')

                for j in range(flags.num_units_in_block):
                    h = slim.conv2d(h, num_outputs=num_filters[i], kernel_size=3, stride=1,
                                    scope='conv' + str(i) + '_' + str(j), padding='SAME', activation_fn=None)

                    if j < (flags.num_units_in_block - 1):
                        h = activation_fn(h, name='activation_' + str(i) + '_' + str(j))
                h = h + shortcut

                h = activation_fn(h, name='activation_' + str(i) + '_' + str(flags.num_units_in_block - 1))
                if i < (len(num_filters) - 1):
                    h = slim.max_pool2d(h, kernel_size=2, stride=2, padding='SAME', scope='max_pool' + str(i))

            if flags.embedding_pooled:
                kernel_size = h.shape.as_list()[-2]
                h = slim.avg_pool2d(h, kernel_size=kernel_size, scope='avg_pool')
            h = slim.flatten(h)
    return h


def get_inception_v3(images, flags, is_training=False, reuse=None, scope=None):
    arg_scope = inception.inception_v3_arg_scope()
    if isinstance(is_training, tf.Variable):
        image_fe_trainable = tf.Variable(flags.image_fe_trainable, trainable=False, name='image_fe_trainable',
                                         dtype=tf.bool)
        is_training_inception = tf.logical_and(is_training, image_fe_trainable)
    else:
        is_training_inception = is_training and flags.image_fe_trainable
    with slim.arg_scope(arg_scope):
        with tf.variable_scope(scope or 'image_feature_extractor', reuse=reuse):
            images = tf.image.resize_bilinear(images, size=[299, 299], align_corners=False)
            scaled_input_tensor = tf.scalar_mul((1.0 / 255), images)
            scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
            scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

            logits, end_points = inception.inception_v3(scaled_input_tensor,
                                                        is_training=is_training_inception, num_classes=1001,
                                                        reuse=reuse)
            h = end_points['PreLogits']
            h = slim.flatten(h)
    return h


def get_encoder(h, flags, is_training, scope="encoder"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        with _get_fc_scope(is_training, flags):
            if flags.hidden_dim > 0:
                h = slim.fully_connected(h, num_outputs=flags.hidden_dim, 
                                         weights_regularizer=tf.contrib.layers.l2_regularizer(scale=flags.weight_decay_fc),
                                         scope='hidden_encoder_layer')
            if flags.latent_dim > 0:
                h = slim.fully_connected(h, num_outputs=flags.latent_dim, 
                                         weights_regularizer=tf.contrib.layers.l2_regularizer(scale=flags.weight_decay_fc),
                                         scope='latent_space_layer')
    return h

                
def get_decoder(h, flags, is_training, scope="decoder"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        with _get_fc_scope(is_training, flags):
            if flags.hidden_dim > 0:
                h = slim.fully_connected(h, num_outputs=flags.hidden_dim, 
                                         weights_regularizer=tf.contrib.layers.l2_regularizer(scale=flags.weight_decay_fc),
                                         scope='hidden_decoder_layer')
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
    :param num_filters:
    :param is_training:
    :param scope:
    :param reuse:
    :return:
    """
    if flags.image_fe_trainable:
        original_shape = images.get_shape().as_list()
        if len(original_shape) == 5:
            images = tf.reshape(images, shape=([-1]+original_shape[2:]))

        num_filters = [round(flags.num_filters * pow(flags.block_size_growth, i)) for i in range(flags.num_blocks)]
        if flags.image_feature_extractor == 'simple_res_net':
            h = get_simple_res_net(images, flags=flags, num_filters=num_filters, is_training=is_training, reuse=reuse,
                                   scope=scope)
        elif flags.image_feature_extractor == 'inception_v3':
            h = get_inception_v3(images, flags=flags, is_training=is_training, reuse=reuse, scope=scope)

        if len(original_shape) == 5:
            h = tf.reshape(h, shape=([-1] + [original_shape[1], h.get_shape().as_list()[-1]]))
    else:
        h = images

    h = tf.reduce_mean(h, axis=1, keepdims=False)
    
    h = get_encoder(h, flags=flags, is_training=is_training, scope="image_encoder")
    
    conv2d_arg_scope, dropout_arg_scope = _get_scope(is_training, flags)
    with conv2d_arg_scope, dropout_arg_scope:
        if flags.dropout:
            h = slim.dropout(h, scope='image_dropout', keep_prob=1.0 - flags.dropout)
            
    h = get_decoder(h, flags=flags, is_training=is_training, scope="image_decoder")            
    return h


def get_2016cnn_bi_lstm(text, text_length, flags, embedding_initializer=None,
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
                                             trainable=flags.word_embed_trainable and is_training,
                                             reuse=tf.AUTO_REUSE,
                                             scope='TextEmbedding')

        h = tf.expand_dims(h, 1)
        conv2d_arg_scope, dropout_arg_scope = _get_scope(is_training, flags)
        with conv2d_arg_scope, dropout_arg_scope:

            h = slim.conv2d(h, num_outputs=flags.num_text_cnn_filt, kernel_size=[1, 3],
                            stride=1, scope='text_conv_1', padding='VALID', activation_fn=activation_fn)
            h = slim.conv2d(h, num_outputs=flags.num_text_cnn_filt, kernel_size=[1, 2],
                            stride=1, scope='text_conv_2', padding='VALID', activation_fn=activation_fn)

            h = slim.max_pool2d(h, kernel_size=[1, 3], stride=[1, 3], padding='VALID',
                                scope='text_max_pool_x3')

            h = slim.conv2d(h, num_outputs=flags.num_text_cnn_filt, kernel_size=[1, 2],
                            stride=1, scope='text_conv_3', padding='VALID', activation_fn=activation_fn)

            text_length = tf.cast(tf.ceil(tf.div(tf.cast(text_length, tf.float32), 3.0)), text_length.dtype)

        h = tf.squeeze(h, [1])

        print(h.get_shape().as_list())
        print(text_length.get_shape().as_list())

        cells_fw = [tf.nn.rnn_cell.LSTMCell(size) for size in [flags.rnn_size]]
        cells_bw = [tf.nn.rnn_cell.LSTMCell(size) for size in [flags.rnn_size]]
        h, *_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw, cells_bw=cells_bw,
                                                               inputs=h, dtype=tf.float32,
                                                               sequence_length=text_length)

        mask = tf.expand_dims(tf.sequence_mask(text_length,
                                               maxlen=h.get_shape().as_list()[1], dtype=tf.float32), axis=-1)
        h = tf.reduce_sum(tf.multiply(h, mask), axis=[1]) / tf.reduce_sum(mask, axis=[1])
        
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
                                             trainable=flags.word_embed_trainable and is_training,
                                             reuse=tf.AUTO_REUSE,
                                             scope='TextEmbedding')
        
        h = tf.expand_dims(h, 1)
        print(h.get_shape().as_list())
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


def get_simple_bi_lstm(text, text_length, flags, embedding_initializer=None,
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
                                             trainable=flags.word_embed_trainable and is_training,
                                             reuse=tf.AUTO_REUSE,
                                             scope='TextEmbedding')

        cells_fw = [tf.nn.rnn_cell.LSTMCell(size) for size in [flags.rnn_size]]
        cells_bw = [tf.nn.rnn_cell.LSTMCell(size) for size in [flags.rnn_size]]
#         initial_states_fw = [cell.zero_state(text.get_shape()[0], dtype=tf.float32) for cell in cells_fw]
#         initial_states_bw = [cell.zero_state(text.get_shape()[0], dtype=tf.float32) for cell in cells_bw]

        h, *_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                               cells_bw=cells_bw,
                                                               inputs=h,
#                                                                initial_states_fw=initial_states_fw,
#                                                                initial_states_bw=initial_states_bw,
                                                               dtype=tf.float32,
                                                               sequence_length=text_length)
        mask = tf.expand_dims(tf.sequence_mask(text_length, maxlen=tf.shape(text)[1], dtype=tf.float32), axis=-1)
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

    if flags.text_feature_extractor == 'simple_bi_lstm':
        h = get_simple_bi_lstm(text, text_length, flags=flags,
                               embedding_initializer=embedding_initializer, is_training=is_training,
                               reuse=reuse, scope=scope)
    elif flags.text_feature_extractor == 'cnn_bi_lstm':
        h = get_cnn_bi_lstm(text, text_length, flags=flags,
                               embedding_initializer=embedding_initializer, is_training=is_training,
                               reuse=reuse, scope=scope)
    elif flags.text_feature_extractor == '2016cnn_bi_lstm':
        h = get_2016cnn_bi_lstm(text, text_length, flags=flags,
                            embedding_initializer=embedding_initializer, is_training=is_training,
                            reuse=reuse, scope=scope)
    
    h = get_encoder(h, flags=flags, is_training=is_training, scope="text_encoder")
        
    conv2d_arg_scope, dropout_arg_scope = _get_scope(is_training, flags)
    with conv2d_arg_scope, dropout_arg_scope:
        if flags.dropout:
            h = slim.dropout(h, scope='text_dropout', keep_prob=1.0 - flags.dropout)

    h = get_decoder(h, flags=flags, is_training=is_training, scope="text_decoder")
    
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


def get_film_layer(h, condition, flags, is_training, scope="film_layer", reuse=tf.AUTO_REUSE):
    """
    :param h: input layer
    :return: conditional batch norm in the form (gamma + 1.0) * h + beta
    """
    activation_fn=ACTIVATION_MAP[flags.activation]
    with tf.variable_scope(scope, reuse=reuse):
        beta_postmultiplier = tf.get_variable(name='beta_postmultiplier', dtype=tf.float32, initializer=0.0,
                                              trainable=True,
                                              regularizer=tf.contrib.layers.l2_regularizer(
                                                  scale=flags.film_weight_decay_postmult,
                                                  scope='penalize_beta'))
        gamma_postmultiplier = tf.get_variable(name='gamma_postmultiplier', dtype=tf.float32, initializer=0.0,
                                               trainable=True,
                                               regularizer=tf.contrib.layers.l2_regularizer(
                                                   scale=flags.film_weight_decay_postmult,
                                                   scope='penalize_gamma'))
        with _get_film_fc_scope(is_training, flags):
            num_filters = h.shape.as_list()[-1]
            beta = slim.fully_connected(condition, num_outputs=num_filters, activation_fn=None, scope='beta')
            gamma = slim.fully_connected(condition, num_outputs=num_filters, activation_fn=None, scope='gamma')

        beta = tf.multiply(beta_postmultiplier, beta, name='postmultiply_beta')
        gamma = 1.0 + tf.multiply(gamma_postmultiplier, gamma, name='postmultiply_gamma')
        
        tf.summary.scalar('beta_weight', beta_postmultiplier)
        tf.summary.scalar('gamma_weight', gamma_postmultiplier)
            
        # h[:,:,None] - Bh x F x 1
        # tf.transpose(gamma) - F x Bc
        # h[:,:,None] * tf.transpose(gamma) - Bh x F x Bc
        # tf.transpose(beta)[None, :, :] - 1 x F x Bc
        # output Bh x F x Bc
        interaction = h[:,:,None] * tf.transpose(gamma) + tf.transpose(beta)[None, :, :]        
        return activation_fn(interaction)


def get_film_interactor(embedding_mod1, embedding_mod2, flags, is_training, scope='film_interactor', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        with _get_fc_scope(is_training, flags):
            embedding_mod1_film = get_film_layer(embedding_mod1, embedding_mod2, flags=flags, 
                                                 is_training=is_training, scope="film_layer_1_to_2", reuse=reuse)
            embedding_mod2_film = get_film_layer(embedding_mod2, embedding_mod1, flags=flags, 
                                                 is_training=is_training, scope="film_layer_2_to_1", reuse=reuse)
            # mod1 embedding has size B1 x F x B2
            # mod1 embedding has size B2 x F x B1
            # This will align the modalities such that both of them have size B1 x F x B2
            embedding_mod2_film = tf.transpose(embedding_mod2_film, perm=[2, 1, 0])
            return embedding_mod1_film, embedding_mod2_film


def get_metric(image_embeddings, text_embeddings, flags, is_training, reuse=False):
    with tf.variable_scope('Metric', reuse=reuse):
        # Here we compute logits of correctly matching text to a given image.
        # We could also compute logits of correctly matching an image to a given text by reversing
        # image_embeddings and text_embeddings
        if flags.modality_interaction == "FILM":
            image_embeddings_film, text_embeddings_film = get_film_interactor(image_embeddings, text_embeddings,
                                                                    flags=flags, is_training=is_training,
                                                                    scope='film_interactor', reuse=reuse)
            logits = -tf.norm(image_embeddings_film - text_embeddings_film, name='neg_euclidian_distance', axis=-2)
        else:
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
        if flags.consistency_loss == "CLASSIFIER":
            labels_class = tf.placeholder(tf.int64, shape=(batch_size_image,), name='class_labels')
        else:
            labels_class = None
        
        return images_placeholder, text_placeholder, text_length_placeholder, labels_txt2img, labels_img2txt, labels_class


def get_images_placeholder(batch_size: int, num_images: int, image_size: int, flags: Namespace):
    if flags.image_fe_trainable:
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_images, image_size, image_size, 3),
                                            name='images')
    else:
        num_features_dict = {'simple_res_net': 512, 'inception_v3': 2048, 'inception_v2': 1024, 'resnet101': 2048}
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
    if flags.image_fe_trainable:
        variables_to_train = tf.trainable_variables()
    else:
        variables_to_train = tf.trainable_variables(scope='(?!.*image_feature_extractor).*')
    # Train operation
    return slim.learning.create_train_op(total_loss=loss, optimizer=optimizer, global_step=global_step,
                                         clip_gradient_norm=flags.clip_gradient_norm,
                                         variables_to_train=variables_to_train)


class NnModelLoader:
    def __init__(self, model_path, batch_size_image, batch_size_text):
        self.batch_size_image = batch_size_image
        self.batch_size_text = batch_size_text

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=os.path.join(model_path, 'train'))
        self.step = int(os.path.basename(latest_checkpoint).split('-')[1])

        flags = Namespace(load_and_save_params(default_params=dict(), exp_dir=model_path))
        self.flags=flags

        with tf.Graph().as_default():
            self.get_metric_input_placeholders()

            self.logits = get_metric(self.image_embeddings, self.text_embeddings,
                                     flags=self.flags, is_training=False, reuse=False)
            
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            print('Loading model')
            global_vars = tf.global_variables()
            if len(global_vars) > 0:
                init_fn = slim.assign_from_checkpoint_fn(latest_checkpoint, global_vars)
                # Run init before loading the weights
                self.sess.run(tf.global_variables_initializer())
                # Load weights
                init_fn(self.sess)

    def get_metric_input_placeholders(self):
        with tf.variable_scope("input"):
            self.image_embeddings = tf.placeholder(shape=(self.batch_size_image, self.flags.embedding_size),
                                                   name='image_embeddings', dtype=tf.float32)
            self.text_embeddings = tf.placeholder(shape=(self.batch_size_text, self.flags.embedding_size),
                                                   name='text_embeddings', dtype=tf.float32)
            
    def predict_batch(self, image_embeddings, text_embeddings):
        feed_dict = {self.image_embeddings: image_embeddings.astype(dtype=np.float32),
                     self.text_embeddings: text_embeddings.astype(dtype=np.float32)}
        return self.sess.run(self.logits, feed_dict)
    
    def predict_all(self, image_embeddings, text_embeddings):
        dist = np.zeros(shape=(image_embeddings.shape[0], text_embeddings.shape[0]))
        num_batches = int(np.ceil(len(image_embeddings) / self.batch_size_image))
        for i in range(num_batches):
            image_embeddings_batch = image_embeddings[i * self.batch_size_image:(i + 1) * self.batch_size_image]
            valid_length = len(image_embeddings_batch)
            if valid_length < self.batch_size_image:
                zeros = np.zeros(shape=(self.batch_size_image-valid_length, image_embeddings.shape[1]))
                image_embeddings_batch = np.concatenate([image_embeddings_batch, zeros], axis=0)
            
            prediction = self.predict_batch(image_embeddings_batch, text_embeddings)
            dist[i * self.batch_size_image:(i + 1) * self.batch_size_image] = prediction[:valid_length]
            
        return dist


class ModelLoader:
    def __init__(self, model_path, batch_size, num_images, num_texts, max_text_len):
        self.batch_size = batch_size
        self.num_images = num_images
        self.num_texts = num_texts
        self.max_text_len = max_text_len

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=os.path.join(model_path, 'train'))
        step = int(os.path.basename(latest_checkpoint).split('-')[1])

        flags = Namespace(load_and_save_params(default_params=dict(), exp_dir=model_path))
        image_size = get_image_size(flags.data_dir)

        with tf.Graph().as_default():
            images_pl, text_pl, text_len_pl, match_labels_txt2img, match_labels_img2txt, _ = get_input_placeholders(
                batch_size_image=batch_size, num_images=num_images, num_texts=num_texts, max_text_len=max_text_len,
                image_size=image_size, flags=flags, scope='inputs')
            if batch_size:
                logits, image_embeddings, text_embeddings = get_inference_graph(
                    images=images_pl, text=text_pl, text_length=text_len_pl, flags=flags, is_training=False)
                self.logits=logits
            else:
                image_embeddings, text_embeddings = get_embeddings(
                    images=images_pl, text=text_pl, text_length=text_len_pl, flags=flags, is_training=False)
            self.images_pl = images_pl
            self.text_pl = text_pl
            self.text_len_pl = text_len_pl
            self.match_labels_txt2img_pl = match_labels_txt2img
            self.match_labels_img2txt_pl = match_labels_img2txt
            self.image_embeddings = image_embeddings
            self.text_embeddings = text_embeddings

            print('Loading model')
            init_fn = slim.assign_from_checkpoint_fn(latest_checkpoint, tf.global_variables())

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # Run init before loading the weights
            self.sess.run(tf.global_variables_initializer())
            # Load weights
            init_fn(self.sess)

            self.flags = flags
            self.step = step

    def predict(self, images, texts, text_len):
        feed_dict = {self.images_pl: images.astype(dtype=np.float32),
                     self.text_pl: texts,
                     self.text_len_pl: text_len}
        return self.sess.run([self.image_embeddings, self.text_embeddings], feed_dict)

    def eval_acc_batch(self, data_set: Dataset, num_samples: int = 100):
        """
        Runs evaluation loop over dataset
        :param data_set:
        :param num_samples: number of tasks to sample from the dataset
        :return:
        """
        num_correct_txt2img = 0.0
        num_correct_img2txt = 0.0
        num_tot = 0.0
        for i in trange(num_samples):
            if self.flags.image_fe_trainable:
                images, texts, text_len, match_labels = data_set.next_batch(
                    batch_size=self.batch_size, num_images=self.flags.num_images, num_texts=self.flags.num_texts)
            else:
                images, texts, text_len, match_labels, _ = data_set.next_batch_features(
                    batch_size=self.batch_size, num_images=self.flags.num_images, num_texts=self.flags.num_texts)

            labels_txt2img, labels_img2txt = match_labels
            feed_dict = {self.images_pl: images.astype(dtype=np.float32),
                         self.text_pl: texts,
                         self.text_len_pl: text_len}
            logits = self.sess.run(self.logits, feed_dict)
            labels_pred_txt2img = np.argmax(logits, axis=-1)
            labels_pred_img2txt = np.argmax(logits, axis=0)

            num_correct_txt2img += sum(labels_pred_txt2img == labels_txt2img)
            num_correct_img2txt += sum(labels_pred_img2txt == labels_img2txt)
            num_tot += len(labels_pred_txt2img)
        return {'acc_txt2img': num_correct_txt2img / num_tot, 'acc_img2txt': num_correct_img2txt / num_tot}
    
    def predict_all(self, data_set, batch_size):
        image_embeddings, text_embeddings = [], []
        if self.flags.image_fe_trainable:
            batch_generator = data_set.sequential_evaluation_batches
        else:
            batch_generator = data_set.sequential_evaluation_batches_features
        for images, texts, text_lengths in tqdm(batch_generator(batch_size=batch_size,
                                                                num_images=self.num_images, num_texts=self.num_texts)):
            image_embeddings_batch, text_embeddings_batch = self.predict(images, texts, text_lengths)
            image_embeddings.append(image_embeddings_batch)
            text_embeddings.append(text_embeddings_batch)
        image_embeddings = np.concatenate(image_embeddings)
        text_embeddings = np.concatenate(text_embeddings)
        return image_embeddings, text_embeddings
    
    def eval_acc(self, data_set: Dataset, batch_size:int):
        """
        Runs evaluation loop over dataset
        :param data_set:
        :return:
        """
        logging.info("Computing embeddings")
        image_embeddings, text_embeddings = self.predict_all(data_set, batch_size)
        logging.info("Computing metrics")
        metrics = ap_at_k_prototypes(support_embeddings=text_embeddings, query_embeddings=image_embeddings,
                                     class_ids=data_set.image_classes, k=50, num_texts=[1, 5, 10, 20, 40, 100])
        return metrics, image_embeddings, text_embeddings
    
    def harmonic_mean(self, x, y):
        return 2*x*y/(x+y)
        
    def eval_acc_gzsh(self, train_loader: Dataset, test_loader: Dataset, batch_size:int):
        """
        Runs evaluation loop in the generalized zero shot learning scenario
        :param data_set:
        :return:
        """
        logging.info("Computing train embeddings")
        image_embeddings_train, text_embeddings_train = self.predict_all(train_loader, batch_size)
        logging.info("Computing test embeddings")
        image_embeddings_test, text_embeddings_test = self.predict_all(test_loader, batch_size)
        
        logging.info("Computing classical zero-shot performance metrics, test")
        metrics_test = ap_at_k_prototypes(support_embeddings=text_embeddings_test, 
                                     query_embeddings=image_embeddings_test,
                                     class_ids=test_loader.image_classes, k=50, num_texts=[1, 5, 10, 20, 40, 100])
        logging.info("Computing classical zero-shot performance metrics, train")
        metrics_train = ap_at_k_prototypes(support_embeddings=text_embeddings_train, 
                                     query_embeddings=image_embeddings_train,
                                     class_ids=train_loader.image_classes, k=50, num_texts=[1, 5, 10, 20, 40, 100])
        
        logging.info("Computing generalized zero-shot performance metrics")
        seen_unseen_text_embeddings = np.concatenate([text_embeddings_train, text_embeddings_test], axis=0)
        seen_unseen_classes = np.concatenate([train_loader.image_classes, test_loader.image_classes], axis=0)
        seen_unseen_subsets = {}
        seen_unseen_subsets['seen'] = list(set(train_loader.image_classes))
        seen_unseen_subsets['unseen'] = list(set(test_loader.image_classes))
        metrics_gzsl_unseen = top1_gzsl(support_embeddings=seen_unseen_text_embeddings, query_embeddings=image_embeddings_test, 
                                 class_ids_support=seen_unseen_classes, class_ids_query=test_loader.image_classes, 
                                 num_texts=[1, 5, 10, 20, 40, 100], seen_unseen_subsets=seen_unseen_subsets)
        metrics_gzsl_seen = top1_gzsl(support_embeddings=seen_unseen_text_embeddings, query_embeddings=image_embeddings_train, 
                                 class_ids_support=seen_unseen_classes, class_ids_query=train_loader.image_classes, 
                                 num_texts=[1, 5, 10, 20, 40, 100], seen_unseen_subsets=seen_unseen_subsets)
        metrics_gzsl = {}
        for key in metrics_gzsl_unseen.keys():
            metrics_gzsl["test_U_"+key] = metrics_gzsl_unseen[key]
            metrics_gzsl["test_S_"+key] = metrics_gzsl_seen[key]
            metrics_gzsl["test_H_"+key] = self.harmonic_mean(metrics_gzsl_unseen[key], metrics_gzsl_seen[key])
        metrics = {}
        for key in metrics_train.keys():
            metrics["train_"+key] = metrics_train[key]
        for key in metrics_test.keys():
            metrics["test_"+key] = metrics_test[key]
            
        metrics.update(metrics_gzsl)
        return metrics, (seen_unseen_text_embeddings, image_embeddings_train, image_embeddings_test)


def eval_acc_batch(flags: Namespace, datasets: Dict[str, Dataset]):
    max_text_len = list(datasets.values())[0].max_text_len
    model = ModelLoader(model_path=flags.pretrained_model_dir, batch_size=flags.eval_batch_size,
                        num_images=flags.num_images, num_texts=flags.num_texts, max_text_len=max_text_len)
    results = {}
    for data_name, dataset in datasets.items():
        results_eval = model.eval_acc_batch(data_set=dataset, num_samples=flags.num_samples_eval)
        for result_name, result_val in results_eval.items():
            results["evaluation_batch_%s/"%(data_name) + result_name] = result_val
            logging.info("accuracy_%s: %.3g" % (result_name + "_" + data_name, result_val))

    log_dir = get_logdir_name(flags)
    eval_writer = summary_writer(log_dir + '/eval')
    eval_writer(model.step, **results)


def eval_acc(flags: Namespace, datasets: Dict[str, Dataset]):
    max_text_len = list(datasets.values())[0].max_text_len
    model = ModelLoader(model_path=flags.pretrained_model_dir, 
                    batch_size=None, num_images=10, num_texts=10, max_text_len=max_text_len)
    
    results_eval, _ = model.eval_acc_gzsh(train_loader=datasets[flags.train_split], 
                                       test_loader=datasets[flags.test_split], batch_size=10)
    for result_name, result_val in results_eval.items():
        logging.info("%s: %.3g" % (result_name, result_val))

    log_dir = get_logdir_name(flags)
    eval_writer = summary_writer(log_dir + '/eval')
    eval_writer(model.step, **results_eval)


def get_image_fe_restorer(flags: Namespace):
    def name_in_checkpoint(var: tf.Variable):
        return '/'.join(var.op.name.split('/')[2:])
        
    if flags.image_feature_extractor == 'inception_v3' and flags.image_fe_trainable:
        vars = tf.get_collection(key=tf.GraphKeys.MODEL_VARIABLES, scope='.*InceptionV3')
        return tf.train.Saver(var_list={name_in_checkpoint(var): var for var in vars})
    elif flags.image_feature_extractor == 'inception_v2' and flags.image_fe_trainable:
        vars = tf.get_collection(key=tf.GraphKeys.MODEL_VARIABLES, scope='.*InceptionV2')
        return tf.train.Saver(var_list={name_in_checkpoint(var): var for var in vars})
    else:
        return None


def test_pretrained_inception_model(images_pl, sess):
    # code to test loaded inception model
    sample_images = ['dog.jpg', 'panda.jpg', 'tinca_tinca.jpg']
    from PIL import Image
    graph = tf.get_default_graph()
    inception_logits_pl = graph.get_tensor_by_name("Model/image_feature_extractor/InceptionV3/Predictions/Reshape_1:0")
    for image in sample_images:
        im = Image.open(image).resize((256, 256))
        im = np.array(im)
        im = im.reshape(-1, 256, 256, 3).astype(np.float32)
        im = np.tile(im, [images_pl.get_shape().as_list()[0], 1, 1, 1])
        logit_values = sess.run(inception_logits_pl, feed_dict={images_pl: im})
        print(image)
        print(np.max(logit_values, axis=-1))
        print(np.argmax(logit_values, axis=-1) - 1)

        
def get_consistency_loss(image_embeddings, text_embeddings, flags, labels=None):
    if flags.mi_weight:
#         # TODO: this can be removed
#         image_embeddings_cond = tf.cond(mi_weight > 0.0, lambda: image_embeddings, 
#                                         lambda: tf.stop_gradient(image_embeddings))
#         text_embeddings_cond = tf.cond(mi_weight > 0.0, lambda: text_embeddings, 
#                                        lambda: tf.stop_gradient(text_embeddings))
#     None, "NMSE", "MI", "CROSSCLASS"

        image_distances = get_dist_mtx(image_embeddings)
        text_distances = get_dist_mtx(text_embeddings)
        
        if flags.consistency_loss == "NMSE":
            consistency_loss = get_rmse_loss(text_distances, image_distances, flags)
        elif flags.consistency_loss == "MI":
            consistency_loss = get_mi_loss(text_distances, image_distances, flags)
        elif flags.consistency_loss == "CROSSCLASS":
            consistency_loss = get_cross_classifier_loss(image_embeddings, text_embeddings, 
                                                         flags, scope="crossclass_loss")
        elif flags.consistency_loss == "SOM":
            consistency_loss = get_som_loss(image_embeddings, text_embeddings, flags)
        elif flags.consistency_loss == "CLASSIFIER":
            consistency_loss = get_classifier_loss(image_embeddings, text_embeddings, flags=flags, labels=labels)
        else:
            consistency_loss = tf.Variable(0.0, trainable=False, 
                                           name='dummy_consistency_loss', dtype=tf.float32)
        tf.summary.scalar('loss/consistency_loss', consistency_loss)
        consistency_loss = consistency_loss
    else:
        consistency_loss = 0.0
    return consistency_loss


def train(flags):
    log_dir = get_logdir_name(flags)
    flags.pretrained_model_dir = log_dir
    log_dir = os.path.join(log_dir, 'train')
    # This is setting to run evaluation loop only once
    flags.max_number_of_evaluations = 1
    flags.eval_interval_secs = 0
    image_size = get_image_size(flags.data_dir)

    # Get datasets
    dataset_splits = get_dataset_splits(dataset_name=flags.dataset, data_dir=flags.data_dir,
                                        splits=[flags.train_split, flags.test_split], flags=flags)
    max_text_len = dataset_splits[flags.train_split].max_text_len
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        is_training = tf.Variable(True, trainable=False, name='is_training', dtype=tf.bool)
        images_pl, text_pl, text_len_pl, match_labels_txt2img_pl, match_labels_img2txt_pl, labels_class = \
            get_input_placeholders(batch_size_image=flags.train_batch_size,
                                   num_images=flags.num_images, num_texts=flags.num_texts,
                                   image_size=image_size, max_text_len=max_text_len,
                                   flags=flags, scope='inputs')

        embedding_initializer = np.zeros(shape=(flags.vocab_size, flags.word_embed_dim), dtype=np.float32)
        vocab = dataset_splits[flags.train_split].word_vectors_idx
        embedding_initializer[:len(vocab)] = vocab
        logits, image_embeddings, text_embeddings = get_inference_graph(images=images_pl, text=text_pl,
                                                                        embedding_initializer=embedding_initializer,
                                                                        text_length=text_len_pl, flags=flags,
                                                                        is_training=True, reuse=False)
        loss_txt2img = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                       labels=tf.one_hot(match_labels_txt2img_pl, flags.train_batch_size)),
            name='loss_txt2img')
        loss_img2txt = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.transpose(logits, perm=[1, 0]),
                                                       labels=tf.one_hot(match_labels_img2txt_pl, flags.train_batch_size)),
            name='loss_img2txt')
        
        mi_weight = tf.Variable(0.0, trainable=False, name='mi_weight', dtype=tf.float32)
        consistency_loss = get_consistency_loss(image_embeddings, text_embeddings, flags, labels=labels_class)
        
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_tot = tf.add_n([0.5 * (1.0-mi_weight) * loss_txt2img, 
                             0.5 * (1.0-mi_weight) * loss_img2txt, 
                             consistency_loss*mi_weight] + regu_losses)
        misclass_txt2img = 1.0 - slim.metrics.accuracy(tf.argmax(logits, 1), match_labels_txt2img_pl)
        misclass_img2txt = 1.0 - slim.metrics.accuracy(tf.argmax(logits, 0), match_labels_img2txt_pl)
        main_train_op = get_main_train_op(loss_tot, global_step, flags)

        tf.summary.scalar('loss/total', loss_tot)
        tf.summary.scalar('loss/txt2img', loss_txt2img)
        tf.summary.scalar('loss/img2txt', loss_img2txt)
        tf.summary.scalar('misclassification/txt2img', misclass_txt2img)
        tf.summary.scalar('misclassification/img2txt', misclass_img2txt)
        summary = tf.summary.merge(tf.get_collection('summaries'))

        # Define session and logging
        summary_writer = tf.summary.FileWriter(log_dir, flush_secs=1)
        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        image_fe_restorer = get_image_fe_restorer(flags=flags)
        supervisor = tf.train.Supervisor(logdir=log_dir, init_feed_dict=None,
                                         summary_op=None,
                                         init_op=tf.global_variables_initializer(),
                                         summary_writer=summary_writer,
                                         saver=saver,
                                         global_step=global_step, save_summaries_secs=flags.save_summaries_secs,
                                         save_model_secs=0)

        with supervisor.managed_session() as sess:
            if image_fe_restorer:
                image_fe_restorer.restore(sess, IMAGE_MODEL_CHECKPOINTS[flags.image_feature_extractor])
                # test_pretrained_inception_model(images_pl, sess)

            checkpoint_step = sess.run(global_step)
            if checkpoint_step > 0:
                checkpoint_step += 1

            loss_tot, dt_train = 0.0, 0.0
            for step in range(checkpoint_step, flags.number_of_steps):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    dt_batch = time.time()
                    if flags.image_fe_trainable:
                        images, text, text_length, match_labels = dataset_splits[flags.train_split].next_batch(
                            batch_size=flags.train_batch_size, num_images=flags.num_images, num_texts=flags.num_texts)
                    else:
                        images, text, text_length, match_labels, class_labels = \
                            dataset_splits[flags.train_split].next_batch_features(
                                batch_size=flags.train_batch_size, 
                                num_images=flags.num_images, num_texts=flags.num_texts)

                    labels_txt2img, labels_img2txt = match_labels
                    dt_batch = time.time() - dt_batch
                    
                    feed_dict = {images_pl: images.astype(dtype=np.float32), text_len_pl: text_length,
                                 text_pl: text,
                                 match_labels_txt2img_pl: labels_txt2img, match_labels_img2txt_pl: labels_img2txt,
                                 is_training: np.random.uniform() < flags.train_bn_proba}
                    if labels_class is not None:
                        feed_dict.update({labels_class: class_labels})

                    if flags.mi_weight and step > flags.mi_train_offset*flags.number_of_steps:
                        feed_dict.update({mi_weight: flags.mi_weight})
                    else:
                        feed_dict.update({mi_weight: -1e-5})

                    if step % 100 == 0:
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()
                        logging.info(
                            "step %d, loss : %.4g, dt: %.3gs, dt_batch: %.3gs" % (step, loss_tot, dt_train, dt_batch))

                    if step % 100 == 0:
                        logits_img2txt = sess.run(logits, feed_dict=feed_dict)
                        logits_img2txt = np.argmax(logits_img2txt, axis=0)
                        num_matches = float(sum(labels_img2txt == logits_img2txt))
                        logging.info("img2txt acc: %.3g" % (num_matches / flags.train_batch_size))

                    t_train = time.time()
                    loss_tot = sess.run(main_train_op, feed_dict=feed_dict)
                    dt_train = time.time() - t_train
                    

                    if step % flags.eval_interval_steps == 0:
                        saver.save(sess, os.path.join(log_dir, 'model'), global_step=step)
                        eval_acc_batch(flags, datasets=dataset_splits)
#                         eval_acc(flags, datasets=dataset_splits)


def test():
    return None





def main(argv=None):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print(os.getcwd())

    default_params = get_arguments()
    log_dir = get_logdir_name(flags=default_params)

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    # This makes sure that we can store a json and recove a namespace back
    flags = Namespace(load_and_save_params(vars(default_params), log_dir))

    if flags.mode == 'train':
        train(flags=flags)
    elif flags.mode == 'eval':
        eval(flags=flags, is_primary=True)
    elif flags.mode == 'test':
        test(flags=flags)


if __name__ == '__main__':
    tf.app.run()