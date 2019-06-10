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
import pathlib
import logging
from common.util import summary_writer
from common.gen_experiments import load_and_save_params
import time
from tqdm import tqdm, trange
from typing import List, Dict, Set
from common.util import Namespace
from datasets import Dataset
from datasets.dataset_list import get_dataset_splits
from common.pretrained_models import IMAGE_MODEL_CHECKPOINTS, get_image_fe_restorer
from model.model import ModelLoader, get_main_train_op, get_input_placeholders, get_inference_graph, \
                        get_consistency_loss, get_image_size


tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)

def get_arguments():
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the data.')
    parser.add_argument('--split_type', type=str, default='GZSL', choices=['ZSL', 'GZSL'],
                        help='Type of the split based on https://arxiv.org/pdf/1703.04394.pdf.')
    parser.add_argument('--train_split', type=str, default='trainval', choices=['train', 'trainval'],
                        help='Split of the data to be used to perform operation.')
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'val'],
                        help='Split of the data to be used to perform operation.')
    parser.add_argument('--dataset', type=str, default='xian2017_cub',
                        choices=['cvpr2016_cub', 'xian2017_cub', 'xian2018_flowers'], help='Dataset to train.')

    # Training parameters
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--number_of_steps', type=int, default=int(150001), help="Number of training steps")
    parser.add_argument('--log_dir', type=str, default='', help='Base log dir')
    parser.add_argument('--exp_dir', type=str, default=None, help='experiement directory for Borgy')
    # Batch parameters
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--num_images', type=int, default=1, help='Number of image samples per image/text pair.')
    parser.add_argument('--num_texts', type=int, default=10, help='Number of text samples per image/text pair.')
    parser.add_argument('--save_summaries_secs', type=int, default=60, help='Time between saving summaries')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--augment', type=bool, default=False)
    # Learning rate paramteres
    parser.add_argument('--init_learning_rate', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--lr_anneal', type=str, default='exp', choices=['exp'])
    parser.add_argument('--n_lr_decay', type=int, default=3)
    parser.add_argument('--lr_decay_rate', type=float, default=10.0)
    parser.add_argument('--clip_gradient_norm', type=float, default=1.0, help='gradient clip norm.')
    parser.add_argument('--weights_initializer_factor', type=float, default=0.1,
                        help='multiplier in the variance of the initialization noise.')
    # Evaluation parameters
    parser.add_argument('--eval_interval_steps', type=int, default=2500,
                        help='Number of train steps between evaluating model in the training loop')
    parser.add_argument('--num_samples_eval', type=int, default=100, help='Number of evaluation samples?')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size?')
    # Test parameters
    parser.add_argument('--pretrained_model_dir', type=str, default='./logs/',
                        help='Path to the pretrained model to run the nearest neigbor baseline test.')
    # Architecture parameters
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'selu', 'swish-1'])
    parser.add_argument('--embedding_size', type=int, default=1024)
    # Image feature extractor
    parser.add_argument('--image_feature_extractor', type=str, default='resnet101',
                        choices=['inception_v3', 'resnet101'], help='Which feature extractor to use')
    # Text feature extractor
    parser.add_argument('--word_embed_dim', type=int, default=300) # this should be equal to the word2vec dimension
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--text_maxlen', type=int, default=30, help='Maximal length of the text description in tokens')
    parser.add_argument('--shuffle_text_in_batch', type=bool, default=False)
    parser.add_argument('--rnn_size', type=int, default=512)
    parser.add_argument('--num_text_cnn_filt', type=int, default=256)
    parser.add_argument('--num_text_cnn_units', type=int, default=3)
    parser.add_argument('--num_text_cnn_blocks', type=int, default=2)
    
    # Cross modal consistency loss
    parser.add_argument('--mi_weight', type=float, default=0.5,
                        help='The weight of the mutual information term between text and image distances')
    parser.add_argument('--consistency_loss', type=str, default="CLASSIFIER", choices=[None, "CLASSIFIER"])
    parser.add_argument('--num_classes_train', type=int, default=250)
    
    parser.add_argument('--txt2img_weight', type=float, default=0.5, help="The weight of the text to image retrieval loss")
    
    args = parser.parse_args()
    print(args)
    return args


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
                  'weight_decay', str(flags.weight_decay), 'image_feature_extractor', str(flags.image_feature_extractor),
                  ]

    if flags.log_dir == '':
        logdir = './logs/' + '-'.join(param_list)
    else:
        logdir = os.path.join(flags.log_dir, '-'.join(param_list))

    if flags.exp_dir is not None:
        # Running a Borgy experiment
        logdir = flags.exp_dir

    return logdir


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
    
    if flags.split_type == "GZSL":
        results_eval, _ = model.eval_acc_gzsh(train_loader=datasets[flags.train_split],
                                              test_loader_unseen=datasets[flags.test_split+"_unseen"],
                                              test_loader_seen=datasets[flags.test_split+"_seen"], 
                                              batch_size=10)
    else:
        raise Exception("NOT IMPLEMENTED")
        
    for result_name, result_val in results_eval.items():
        logging.info("%s: %.3g" % (result_name, result_val))

    log_dir = get_logdir_name(flags)
    eval_writer = summary_writer(log_dir + '/eval')
    eval_writer(model.step, **results_eval)


def load_data(flags):
    if flags.split_type == "ZSL":
        dataset_splits = get_dataset_splits(dataset_name=flags.dataset, data_dir=flags.data_dir,
                                            splits=[flags.train_split, flags.test_split], flags=flags)
    elif flags.split_type == "GZSL":
        test_splits = [flags.test_split+"_seen", flags.test_split+"_unseen"]
        dataset_splits = get_dataset_splits(dataset_name=flags.dataset, data_dir=flags.data_dir,
                                            splits=[flags.train_split] + test_splits, flags=flags)
    return dataset_splits

    
def train(flags):
    log_dir = get_logdir_name(flags)
    flags.pretrained_model_dir = log_dir
    log_dir = os.path.join(log_dir, 'train')
    image_size = get_image_size(flags.data_dir)

    # Get datasets
    dataset_splits = load_data(flags)
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
        loss_tot = tf.add_n([flags.txt2img_weight * (1.0-mi_weight) * loss_txt2img, 
                             (1.0 - flags.txt2img_weight) * (1.0-mi_weight) * loss_img2txt, 
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
                
            checkpoint_step = sess.run(global_step)
            if checkpoint_step > 0:
                checkpoint_step += 1

            loss_tot, dt_train = 0.0, 0.0
            for step in range(checkpoint_step, flags.number_of_steps):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    dt_batch = time.time()
                    images, text, text_length, match_labels, class_labels = \
                        dataset_splits[flags.train_split].next_batch_features(
                                batch_size=flags.train_batch_size, 
                                num_images=flags.num_images, num_texts=flags.num_texts)
                    labels_txt2img, labels_img2txt = match_labels
                    dt_batch = time.time() - dt_batch
                    
                    feed_dict = {images_pl: images.astype(dtype=np.float32), text_len_pl: text_length,
                                 text_pl: text,
                                 match_labels_txt2img_pl: labels_txt2img, match_labels_img2txt_pl: labels_img2txt,
                                 is_training: True}
                    if labels_class is not None:
                        feed_dict.update({labels_class: class_labels})

                    if flags.mi_weight:
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
                        eval_acc(flags, datasets=dataset_splits)


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
    train(flags=flags)


if __name__ == '__main__':
    tf.app.run()