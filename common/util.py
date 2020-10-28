import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import os
import fnmatch


def summary_writer(log_dir):
    """Convenient wrapper for writing summaries."""
    writer = tf.summary.FileWriter(log_dir)

    def call(step, **value_dict):
        summary = tf.Summary()
        for tag, value in value_dict.items():
            summary.value.add(tag=tag, simple_value=value)
        writer.add_summary(summary, step)
        writer.flush()
    return call


ACTIVATION_MAP = {"relu": tf.nn.relu,
                  "selu": tf.nn.selu,
                  "swish-1": lambda x, name='swish-1': tf.multiply(x, tf.nn.sigmoid(x), name=name),
                  }


class Namespace(object):
    """
    Wrapper around dictionary to make it saveable
    """
    def __init__(self, adict):
        self.__dict__.update(adict)
