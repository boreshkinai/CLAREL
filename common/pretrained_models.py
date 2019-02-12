import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import numpy as np
import os
import tarfile
import argparse
import pathlib
import urllib.request as request
from keras.utils import get_file
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import variable_scope

slim = tf.contrib.slim

INCEPTION_V3_PATH = os.path.join(os.sep, 'mnt', 'datasets', 'public', 'research', 'cvpr2016_cub', 'inception_v3.ckpt')
INCEPTION_V3_URL = "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
INCEPTION_V2_PATH = os.path.join(os.sep, 'mnt', 'datasets', 'public', 'research', 'cvpr2016_cub', 'inception_v2.ckpt')
INCEPTION_V2_URL = "http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz"

class InceptionLoader:
    
    def __init__(self):
        pass
    
    def __del__(self):
        tf.reset_default_graph()
        self.sess.close()

    def predict(self, images):
        predict_values, logit_values, embedding = self.sess.run(
            [self.end_points['Predictions'], self.logits, self.end_points['PreLogits']],
            feed_dict={self.input_tensor: images})
        embedding = np.reshape(embedding, (embedding.shape[0], embedding.shape[-1]))
        return predict_values, logit_values, embedding
    
    def _get_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
    def _restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, self.checkpoint_file)
        
    def _get_inception_preprocessing(self, inception_size):
        self.input_tensor = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, 3), name='input_image')
        input_tensor = tf.image.resize_bilinear(self.input_tensor, size=[inception_size, inception_size], align_corners=False)
        scaled_input_tensor = tf.scalar_mul((1.0 / 255), input_tensor)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        self.scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
        

class InceptionV3Loader(InceptionLoader):

    def __init__(self, model_path=INCEPTION_V3_PATH, image_size=299):
        cache_dir = os.sep.join(model_path.split(os.sep)[:-1])
        get_file(fname=model_path.split(os.sep)[-1], cache_dir=cache_dir, cache_subdir='',
                 origin=INCEPTION_V3_URL, untar=True)
        
        self._get_session()
        self.checkpoint_file = model_path
        self.image_size = image_size
        self._get_inception_preprocessing(inception_size=299)

        self.arg_scope = inception.inception_v3_arg_scope()
        with slim.arg_scope(self.arg_scope):
            self.logits, self.end_points = inception.inception_v3(
                self.scaled_input_tensor, is_training=False, num_classes=1001, reuse=False)
        self._restore()
        

class InceptionV2Loader(InceptionLoader):

    def __init__(self, model_path=INCEPTION_V2_PATH, image_size=299):
        cache_dir = os.sep.join(model_path.split(os.sep)[:-1])
        get_file(fname=model_path.split(os.sep)[-1], cache_dir=cache_dir, cache_subdir='',
                 origin=INCEPTION_V2_URL, untar=True)
        
        self._get_session()
        self.checkpoint_file = model_path
        self.image_size = image_size
        self._get_inception_preprocessing(inception_size=224)

        self.arg_scope = inception.inception_v2_arg_scope()
        with slim.arg_scope(self.arg_scope):
            self.logits, self.end_points = inception.inception_v2(
                self.scaled_input_tensor, is_training=False, num_classes=1001, reuse=False)
            end_point = list(self.end_points.values())[-3]
            kernel_size = end_point.get_shape().as_list()[-2]
            with variable_scope.variable_scope('InceptionV2/Logits', reuse=True):
                embedding = layers_lib.avg_pool2d(end_point, kernel_size, padding='VALID',
                                                  scope='AvgPool_1a_%dx%d'%(kernel_size, kernel_size))
            self.end_points['PreLogits'] = embedding
        self._restore()


IMAGE_MODELS = {"inception_v3": InceptionV3Loader, "inception_v2": InceptionV2Loader}
IMAGE_MODEL_CHECKPOINTS = {"inception_v3": INCEPTION_V3_PATH, "inception_v2": INCEPTION_V2_PATH}
