#!/usr/bin/env python3

"""Training and evaluation entry point."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import tensorflow as tf
import pathlib
import logging
from common.util import summary_writer
from common.gen_experiments import load_and_save_params
import time
from typing import List, Dict, Set
from common.util import Namespace
from datasets import Dataset
from datasets.dataset_list import get_dataset_splits
from model.model_loader import ModelLoader
from model.model import Model

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
    parser.add_argument('--augment', type=bool, default=False)
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
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
    
    # Classifier loss
    parser.add_argument('--kappa', type=float, default=0.5,
                        help='The weight of the mutual information term between text and image distances')
    parser.add_argument('--num_classes_train', type=int, default=250)
    
    parser.add_argument('--lambdaa', type=float, default=0.5, help="The weight of the text to image retrieval loss")
    
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
    model = ModelLoader(model_path=flags.model_path, batch_size=flags.eval_batch_size,
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
    model = ModelLoader(model_path=flags.model_path, 
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
    flags.model_path = log_dir
    log_dir = os.path.join(log_dir, 'train')
    #
    # Get datasets
    #
    dataset_splits = load_data(flags)
    max_text_len = dataset_splits[flags.train_split].max_text_len
    with tf.Graph().as_default():

        embedding_initializer = np.zeros(shape=(flags.vocab_size, flags.word_embed_dim), dtype=np.float32)
        vocab = dataset_splits[flags.train_split].word_vectors_idx
        embedding_initializer[:len(vocab)] = vocab
        #
        # Build model graph
        #
        model = Model(flags=flags, is_training=True, embedding_initializer=embedding_initializer)
        model.get_input_placeholders(batch_size_image=flags.train_batch_size, num_images=flags.num_images,
                                     num_texts=flags.num_texts, max_text_len=max_text_len, scope='inputs')
        model.get_inference_graph(reuse=False)
        model.get_losses()
        main_train_op = model.get_main_train_op()
        summary = model.get_summaries()
        #
        # Define session and logging
        #
        summary_file_writer = tf.summary.FileWriter(log_dir, flush_secs=1)
        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        supervisor = tf.train.Supervisor(logdir=log_dir, init_feed_dict=None,
                                         summary_op=None,
                                         init_op=tf.global_variables_initializer(),
                                         summary_writer=summary_file_writer,
                                         saver=saver,
                                         global_step=model.global_step, save_summaries_secs=flags.save_summaries_secs,
                                         save_model_secs=0)
        with supervisor.managed_session() as sess:
            #
            # Main training loop
            #
            checkpoint_step = sess.run(model.global_step)
            if checkpoint_step > 0:
                checkpoint_step += 1

            loss_tot, dt_train = 0.0, 0.0
            for step in range(checkpoint_step, flags.number_of_steps):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    #
                    # Sample batch
                    #
                    dt_batch = time.time()
                    images, text, text_length, match_labels, class_labels = \
                        dataset_splits[flags.train_split].next_batch_features(
                                batch_size=flags.train_batch_size, 
                                num_images=flags.num_images, num_texts=flags.num_texts)
                    labels_txt2img, labels_img2txt = match_labels
                    dt_batch = time.time() - dt_batch
                    
                    feed_dict = {model.images: images.astype(dtype=np.float32),
                                 model.text_length: text_length, model.text: text,
                                 model.labels_txt2img: labels_txt2img, model.labels_img2txt: labels_img2txt,
                                 model.labels_class: class_labels}
                    #
                    # Run summaries and logging
                    #
                    if step % 100 == 0:
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_file_writer.add_summary(summary_str, step)
                        summary_file_writer.flush()
                        logging.info(
                            "step %d, loss : %.4g, dt: %.3gs, dt_batch: %.3gs" % (step, loss_tot, dt_train, dt_batch))
                        logits_text_retrieval = sess.run(model.logits, feed_dict=feed_dict)
                        logits_text_retrieval = np.argmax(logits_text_retrieval, axis=0)
                        num_matches = float(sum(labels_img2txt == logits_text_retrieval))
                        logging.info("text retrieval acc: %.3g" % (num_matches / flags.train_batch_size))
                    #
                    # Run main train operation
                    #
                    t_train = time.time()
                    loss_tot = sess.run(main_train_op, feed_dict=feed_dict)
                    dt_train = time.time() - t_train
                    #
                    # Run evaluation and save model
                    #
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