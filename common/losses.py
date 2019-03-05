import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim



def get_entropy_kde(x, h):
    N, d = x.shape.as_list()
    
    x = tf.expand_dims(x, -1)
    diff = (x-tf.transpose(x))
    diff = -tf.square(diff)
    diff = tf.reduce_sum(diff, axis=len(x.shape)-2) / (2.0*h**2)
    
    hx = -tf.reduce_mean(tf.reduce_logsumexp(diff, axis=-1), keep_dims=False, name="entropy")
    hx += np.log(N) + np.log(2*np.pi*h**2) * d / 2.0
    return hx


def get_mi_loss(modality1_dist, modality2_dist, flags):
    
    lower_tril_idxs = np.column_stack(np.tril_indices(n=flags.train_batch_size, k=-1))
    x = tf.expand_dims(tf.gather_nd(modality1_dist, lower_tril_idxs), axis=-1) 
    y = tf.expand_dims(tf.gather_nd(modality2_dist, lower_tril_idxs), axis=-1)
    z = tf.concat([x, y], axis=1)
    
    hx = get_entropy_kde(x, flags.mi_kernel_width)
    hy = get_entropy_kde(y, flags.mi_kernel_width)
    hxy = get_entropy_kde(z, flags.mi_kernel_width)
    
    mi = -hxy + hx + hy
    perplexity = tf.pow(2.0, hx*hy)
    return -tf.div_no_nan(mi, perplexity)


def get_dist_mtx(x):
    diff = (tf.expand_dims(x, -1)-tf.transpose(x))
    diff = -tf.square(diff)
    diff = tf.reduce_mean(diff, axis=len(x.shape)-1)
    return diff


def get_dist_mtx_xy(x, y):
    diff = (tf.expand_dims(x, -1)-y)
    diff = -tf.square(diff)
    diff = tf.reduce_mean(diff, axis=len(x.shape)-1)
    return diff


def get_rmse_loss(modality1_dist, modality2_dist, flags):
    lower_tril_idxs = np.column_stack(np.tril_indices(n=flags.train_batch_size, k=-1))
    x = tf.expand_dims(tf.gather_nd(modality1_dist, lower_tril_idxs), axis=-1) 
    y = tf.expand_dims(tf.gather_nd(modality2_dist, lower_tril_idxs), axis=-1)
    y_const = tf.stop_gradient(y)
    x_const = tf.stop_gradient(x)
    loss1 = tf.div_no_nan( tf.square(tf.norm(x - y_const)), tf.norm(x) * tf.norm(y_const) )
    loss2 = tf.div_no_nan( tf.square(tf.norm(x_const - y)), tf.norm(x_const) * tf.norm(y) )
    return 0.5*loss1 + 0.5*loss2


@tf.custom_gradient
def grad_invert(x):
    def grad(dy):
        return dy
    return x, grad


def get_cross_classifier_loss(features_modality1, features_modality2, flags, scope="cluster_loss"):
    num_features = features_modality1.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        class_prototypes = tf.get_variable(name='class_prototypes', shape=(flags.cross_class_num_clusters, num_features), 
                                           dtype=tf.float32, trainable=True, regularizer=None,
                                           initializer=tf.initializers.random_uniform)
        class_prototypes_norm = tf.norm(class_prototypes, axis=-1, keepdims=True)
        class_prototypes = tf.div_no_nan(class_prototypes, class_prototypes_norm)
        
        class_prototypes_adversarial = class_prototypes
        
        tf.summary.scalar('class_prototypes/min_norm_prototypes', tf.reduce_min(class_prototypes_norm))
        tf.summary.scalar('class_prototypes/max_norm_prototypes', tf.reduce_max(class_prototypes_norm))
        
        features_modality1_norm = tf.norm(features_modality1, axis=-1, keepdims=True)
        features_modality2_norm = tf.norm(features_modality2, axis=-1, keepdims=True)
        features_modality1 = tf.div_no_nan(features_modality1, features_modality1_norm)
        features_modality2 = tf.div_no_nan(features_modality2, features_modality2_norm)
        
        tf.summary.scalar('class_prototypes/min_norm_mod1', tf.reduce_min(features_modality1_norm))
        tf.summary.scalar('class_prototypes/max_norm_mod1', tf.reduce_max(features_modality1_norm))
        tf.summary.scalar('class_prototypes/min_norm_mod2', tf.reduce_min(features_modality2_norm))
        tf.summary.scalar('class_prototypes/max_norm_mod2', tf.reduce_max(features_modality2_norm))
        
        logits_modality1 = 100.0*get_dist_mtx_xy(features_modality1, tf.transpose(class_prototypes))
        logits_modality2 = 100.0*get_dist_mtx_xy(features_modality2, tf.transpose(class_prototypes))
        
        logits_modality1_adversarial = 100.0*get_dist_mtx_xy(tf.stop_gradient(features_modality1), 
                                                             tf.transpose(class_prototypes_adversarial))
        logits_modality2_adversarial = 100.0*get_dist_mtx_xy(tf.stop_gradient(features_modality2), 
                                                             tf.transpose(class_prototypes_adversarial))
        
        tf.summary.scalar('class_prototypes/min_logits_adversarial_mod1', tf.reduce_min(logits_modality1_adversarial))
        tf.summary.scalar('class_prototypes/max_logits_adversarial_mod1', tf.reduce_max(logits_modality1_adversarial))
        
        classes_modality1 = tf.argmax(tf.stop_gradient(logits_modality1_adversarial), axis=-1)
        classes_modality2 = tf.argmax(tf.stop_gradient(logits_modality2_adversarial), axis=-1)
        
        loss1 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classes_modality1, logits=logits_modality2))
        loss2 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classes_modality2, logits=logits_modality1))
        
        accuracy_mod1_ref = slim.metrics.accuracy(tf.argmax(logits_modality2, -1), classes_modality1)
        accuracy_mod2_ref = slim.metrics.accuracy(tf.argmax(logits_modality1, -1), classes_modality2)

        tf.summary.scalar('class_prototypes/accuracy_mod1_ref', accuracy_mod1_ref)
        tf.summary.scalar('class_prototypes/accuracy_mod2_ref', accuracy_mod2_ref)
        
    return 0.5*loss1 + 0.5*loss2
    
    




