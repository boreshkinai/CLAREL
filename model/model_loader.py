import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
from common.gen_experiments import load_and_save_params
from tqdm import tqdm, trange
from common.util import Namespace
from datasets import Dataset
from common.metrics import ap_at_k_prototypes, top1_gzsl
from model.model import Model


tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)


class MetricLoader:
    def __init__(self, model_path, batch_size_image, batch_size_text):
        self.batch_size_image = batch_size_image
        self.batch_size_text = batch_size_text

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=os.path.join(model_path, 'train'))
        self.step = int(os.path.basename(latest_checkpoint).split('-')[1])

        flags = Namespace(load_and_save_params(default_params=dict(), exp_dir=model_path))
        self.flags=flags

        with tf.Graph().as_default():
            self.get_metric_input_placeholders()
            model = Model(flags=flags, is_training=False)
            # - is because we use negative distance for logits
            self.logits = -model.get_metric(self.image_embeddings, self.text_embeddings)
            
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

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
        self.model_path = model_path
        
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=os.path.join(model_path, 'train'))
        step = int(os.path.basename(latest_checkpoint).split('-')[1])

        flags = Namespace(load_and_save_params(default_params=dict(), exp_dir=model_path))
        with tf.Graph().as_default():

            model = Model(flags=flags, is_training=False)
            model.get_input_placeholders(batch_size_image=batch_size, num_images=num_images,
                                         num_texts=num_texts, max_text_len=max_text_len, scope='inputs')
            if batch_size:
                model.get_inference_graph()
                self.logits=model.logits
            else:
                model.get_embeddings()
            self.images_pl = model.images
            self.text_pl = model.text
            self.text_len_pl = model.text_length
            self.match_labels_txt2img_pl = model.labels_txt2img
            self.match_labels_img2txt_pl = model.labels_txt2img
            self.image_embeddings = model.image_embeddings
            self.text_embeddings = model.text_embeddings

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
        
    def eval_acc_gzsh(self, train_loader: Dataset, test_loader_unseen: Dataset, test_loader_seen: Dataset, 
                      batch_size:int, seen_adjustment=0.0):
        """
        Runs evaluation loop in the generalized zero shot learning scenario
        :param data_set:
        :return:
        """
        logging.info("Computing train embeddings")
        image_embeddings_train, text_embeddings_train = self.predict_all(train_loader, batch_size)
        logging.info("Computing test embeddings, unseen")
        image_embeddings_test_unseen, text_embeddings_test_unseen = self.predict_all(test_loader_unseen, batch_size)
        logging.info("Computing test embeddings, seen")
        image_embeddings_test_seen, text_embeddings_test_seen = self.predict_all(test_loader_seen, batch_size)
        
        logging.info("Computing generalized zero-shot performance metrics")        
        seen_unseen_text_embeddings = np.concatenate(
            [text_embeddings_train, text_embeddings_test_unseen, text_embeddings_test_seen], axis=0)
        seen_unseen_classes = np.concatenate(
            [train_loader.image_classes, test_loader_unseen.image_classes, test_loader_seen.image_classes], axis=0)
        seen_unseen_subsets = {}
        seen_unseen_subsets['seen'] = list(set(train_loader.image_classes))
        seen_unseen_subsets['unseen'] = list(set(test_loader_unseen.image_classes))
        
        metric_model = MetricLoader(model_path=self.model_path, batch_size_image=100, 
                                    batch_size_text=len(set(seen_unseen_classes)))
        
        metrics_gzsl_unseen = top1_gzsl(support_embeddings=seen_unseen_text_embeddings, 
                                        query_embeddings=image_embeddings_test_unseen, 
                                        class_ids_support=seen_unseen_classes, class_ids_query=test_loader_unseen.image_classes, 
                                        num_texts=[1, 5, 10, 20, 40, 100], seen_unseen_subsets=seen_unseen_subsets,
                                        distance_metric=metric_model, seen_adjustment=seen_adjustment)
        
        metrics_gzsl_seen = top1_gzsl(support_embeddings=seen_unseen_text_embeddings, 
                                      query_embeddings=image_embeddings_test_seen, 
                                      class_ids_support=seen_unseen_classes, class_ids_query=test_loader_seen.image_classes, 
                                      num_texts=[1, 5, 10, 20, 40, 100], seen_unseen_subsets=seen_unseen_subsets,
                                      distance_metric=metric_model, seen_adjustment=seen_adjustment)
        
        logging.info("Computing classical zero-shot performance metrics, test")
        metric_model = MetricLoader(model_path=self.model_path, batch_size_image=100, 
                                    batch_size_text=len(set(test_loader_unseen.image_classes)))
        metrics_test = ap_at_k_prototypes(support_embeddings=text_embeddings_test_unseen, 
                                          query_embeddings=image_embeddings_test_unseen,
                                          class_ids=test_loader_unseen.image_classes, k=50, num_texts=[1, 5, 10, 20, 40, 100],
                                          distance_metric_prototypes=metric_model)
        logging.info("Computing classical zero-shot performance metrics, train")
        metric_model = MetricLoader(model_path=self.model_path, batch_size_image=100, 
                                    batch_size_text=len(set(train_loader.image_classes)))
        metrics_train = ap_at_k_prototypes(support_embeddings=text_embeddings_train, 
                                           query_embeddings=image_embeddings_train,
                                           class_ids=train_loader.image_classes, k=50, num_texts=[1, 5, 10, 20, 40, 100],
                                           distance_metric_prototypes=metric_model)
        
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
        embeddings = (seen_unseen_text_embeddings, image_embeddings_train, image_embeddings_test_seen, image_embeddings_test_unseen)
        return metrics, embeddings
    
    