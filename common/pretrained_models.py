import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import numpy as np
import os
import tarfile
import argparse
import pathlib
import urllib.request as request
from keras.utils import get_file

slim = tf.contrib.slim

INCEPTION_V3_PATH = os.path.join(os.sep, 'mnt', 'datasets', 'public', 'research', 'cvpr2016_cub', 'inception_v3.ckpt')
INCEPTION_V3_URL = "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
INCEPTION_V2_PATH = os.path.join(os.sep, 'mnt', 'datasets', 'public', 'research', 'cvpr2016_cub', 'inception_v2.ckpt')
INCEPTION_V2_URL = "http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz"


class InceptionV3Loader:

    def __init__(self, model_path=INCEPTION_V3_PATH, image_size=299):
        get_file(INCEPTION_V3_PATH, INCEPTION_V3_URL, untar=True)
        self.sess = tf.Session()
        self.checkpoint_file = model_path

        self.input_tensor = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3), name='input_image')
        input_tensor = tf.image.resize_bilinear(self.input_tensor, size=[299, 299], align_corners=False)
        scaled_input_tensor = tf.scalar_mul((1.0 / 255), input_tensor)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        self.scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

        self.arg_scope = inception.inception_v3_arg_scope()
        with slim.arg_scope(self.arg_scope):
            self.logits, self.end_points = inception.inception_v3(
                scaled_input_tensor, is_training=False, num_classes=1001, reuse=False)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.checkpoint_file)

    def predict(self, images):
        predict_values, logit_values, embedding = self.sess.run(
            [self.end_points['Predictions'], self.logits, self.end_points['PreLogits']],
            feed_dict={self.input_tensor: images})
        embedding = np.reshape(embedding, (embedding.shape[0], embedding.shape[-1]))
        return predict_values, logit_values, embedding


class InceptionV2Loader:

    def __init__(self, model_path=INCEPTION_V2_PATH, image_size=299):
        get_file(INCEPTION_V2_PATH, INCEPTION_V2_URL, untar=True)
        self.sess = tf.Session()
        self.checkpoint_file = model_path

        self.input_tensor = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3), name='input_image')
        input_tensor = tf.image.resize_bilinear(self.input_tensor, size=[224, 224], align_corners=False)
        scaled_input_tensor = tf.scalar_mul((1.0 / 255), input_tensor)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        self.scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

        self.arg_scope = inception.inception_v2_arg_scope()
        with slim.arg_scope(self.arg_scope):
            self.logits, self.end_points = inception.inception_v2(
                scaled_input_tensor, is_training=False, num_classes=1001, reuse=False)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.checkpoint_file)

    def predict(self, images):
        predict_values, logit_values, embedding = self.sess.run(
            [self.end_points['Predictions'], self.logits, self.end_points['PreLogits']],
            feed_dict={self.input_tensor: images})
        embedding = np.reshape(embedding, (embedding.shape[0], embedding.shape[-1]))
        return predict_values, logit_values, embedding


IMAGE_MODELS = {"inception_v3": InceptionV3Loader, "inception_v2": InceptionV2Loader}
