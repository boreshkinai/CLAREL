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
        return -dy
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


def get_classifier_loss(features_modality1, features_modality2, labels, flags, scope="classifier_loss"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        samples = tf.concat([features_modality1, features_modality2], axis=0)
        labels = tf.concat([labels, labels], axis=0)
        logits = slim.fully_connected(samples, num_outputs=flags.num_classes_train, activation_fn=None, 
                                      normalizer_fn=None, trainable=True, weights_regularizer=None, 
                                      scope='classifier_fc')
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits_modality2))
    return loss
    

def get_som_loss(features_modality1, features_modality2, flags, scope="som_loss"):
    # https://en.wikipedia.org/wiki/Self-organizing_map
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.700.1701&rep=rep1&type=pdf
    num_features = features_modality1.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Normalize features
        features_modality1_norm = tf.stop_gradient(tf.norm(features_modality1, axis=-1, keepdims=True) + 1e-10)
        features_modality2_norm = tf.stop_gradient(tf.norm(features_modality2, axis=-1, keepdims=True) + 1e-10)
#         features_modality1 = tf.div_no_nan(features_modality1, features_modality1_norm)
#         features_modality2 = tf.div_no_nan(features_modality2, features_modality2_norm)
        
        features_modality1 = features_modality1 / 35.0
        features_modality2 = features_modality2 / 35.0
        # Concatenate text and image modalities
#         d_samples = tf.concat([features_modality1, features_modality2], axis=0)
        d_samples = 0.5*(features_modality1 + features_modality2)
        # Implement SOM
        with tf.variable_scope("SOM", reuse=tf.AUTO_REUSE):
            class_prototypes = tf.get_variable(name='class_prototypes', 
                                               shape=(flags.cross_class_num_clusters, num_features), 
                                               dtype=tf.float32, trainable=False, regularizer=None,
                                               initializer=tf.initializers.random_uniform(maxval=0.05))
            
            assert int(np.sqrt(flags.cross_class_num_clusters))**2 == flags.cross_class_num_clusters
            
            som_grid = np.meshgrid(np.arange(np.sqrt(flags.cross_class_num_clusters)), 
                                   np.arange(np.sqrt(flags.cross_class_num_clusters)))
            som_grid = np.array(som_grid).reshape((2, -1)).transpose()
            som_dist = np.sum(np.square(np.expand_dims(som_grid, -1) - som_grid.transpose()), axis=1)
            sigma = flags.cross_class_sigma_0
            H = np.exp(-som_dist/(2.0*sigma)).astype(np.float32)
            H = (-som_dist/(2.0*sigma)).astype(np.float32)
            
            sample_prototype_distances = -tf.reduce_mean(tf.square(
                tf.expand_dims(d_samples, -1) - tf.transpose(class_prototypes)), axis=1)
            bmu_idxs = tf.argmax(sample_prototype_distances, axis=-1, output_type=tf.int32)

            h_cj_i = tf.gather(H, bmu_idxs)
#             bmu_prototype_weights = h_cj_i / (tf.reduce_sum(h_cj_i, axis=0, keepdims=True) + 1e-50)
            bmu_prototype_weights = tf.nn.softmax(h_cj_i, axis=0)

            new_prototypes = tf.matmul(bmu_prototype_weights, d_samples, transpose_a=True)
            update_step = (1-flags.cross_class_decay)

            class_prototypes_update_op = tf.assign(class_prototypes,
                                                   class_prototypes + \
                                                   update_step * (new_prototypes - class_prototypes))
        
        logits_modality1 = -tf.sqrt(-get_dist_mtx_xy(features_modality1, tf.transpose(class_prototypes)))
        logits_modality1 *= flags.cross_class_metric_scale
        logits_modality2 = -tf.sqrt(-get_dist_mtx_xy(features_modality2, tf.transpose(class_prototypes)))
        logits_modality2 *= flags.cross_class_metric_scale
                
        classes_modality1 = tf.argmax(tf.stop_gradient(logits_modality1), axis=-1)
        classes_modality2 = tf.argmax(tf.stop_gradient(logits_modality2), axis=-1)
        
        # Implement classifier
        with tf.control_dependencies([class_prototypes_update_op]):
            loss1 = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classes_modality1, logits=logits_modality2))
            loss2 = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classes_modality2, logits=logits_modality1))
        
        accuracy_mod1_ref = slim.metrics.accuracy(tf.argmax(logits_modality2, -1), classes_modality1)
        accuracy_mod2_ref = slim.metrics.accuracy(tf.argmax(logits_modality1, -1), classes_modality2)
        
        tf.summary.scalar('class_prototypes/accuracy_mod1_ref', accuracy_mod1_ref)
        tf.summary.scalar('class_prototypes/accuracy_mod2_ref', accuracy_mod2_ref)
        
        class_prototypes_norm = tf.norm(class_prototypes, axis=-1, keepdims=True)
        tf.summary.scalar('class_prototypes/min_norm_prototypes', tf.reduce_min(class_prototypes_norm))
        tf.summary.scalar('class_prototypes/max_norm_prototypes', tf.reduce_max(class_prototypes_norm))
        tf.summary.scalar('class_prototypes/min_norm_mod1', tf.reduce_min(logits_modality1))
        tf.summary.scalar('class_prototypes/max_norm_mod1', tf.reduce_max(logits_modality1))
        tf.summary.scalar('class_prototypes/min_norm_mod2', tf.reduce_min(logits_modality2))
        tf.summary.scalar('class_prototypes/max_norm_mod2', tf.reduce_max(logits_modality2))
        tf.summary.scalar('class_prototypes/loss1', loss1)
        tf.summary.scalar('class_prototypes/loss2', loss2)
        
        
#         # Concatenate text and image modalities
#         d_samples = features_modality1
# #         tf.concat([features_modality1, features_modality2], axis=0)
#         # For each sample in the data batch, find the best matching prototype
#         sample_prototype_distances = get_dist_mtx_xy(d_samples, tf.transpose(class_prototypes))
#         # This is the indexes of class prototypes best matching to a given data point
#         bmu_idxs = tf.argmax(sample_prototype_distances, axis=-1, output_type=tf.int32)
#         # This is the indexes of data points best matching to a given  class prototype
#         bm_data_idxs = tf.argmax(sample_prototype_distances, axis=0, output_type=tf.int32)
#         data_idx_range = tf.range(tf.shape(d_samples)[0])
#         # This is the set of BMUs
#         bmus = tf.gather(class_prototypes, bmu_idxs)
# #         # This is the truth table showing true when a given datapoint is the closest to a BMU 
# #         best_data_point_conditon = tf.equal(tf.gather(bm_data_idxs, bmu_idxs), data_idx_range)
# #         # Only those BMUs in bmus have non-zero update that correspond to the closest data point
# #         data_minus_bmu = (1 - flags.cross_class_decay) * (d_samples - bmus)
# #         bmus_update = tf.where(best_data_point_conditon, 
# #                                x=data_minus_bmu, y=tf.zeros_like(d_samples), name="select_bmu_update")
#         data_minus_bmu = (1 - flags.cross_class_decay) * (d_samples - bmus)
#         bmu_idxs_unique, bmu_idxs_position, bmu_idxs_unique_count = tf.unique_with_counts(bmu_idxs)
#         bmu_inverse_weights = tf.gather(bmu_idxs_unique_count, bmu_idxs_position)
#         # Divide by the number of data points per BMU
#         bmus_update = tf.div(data_minus_bmu, tf.cast(bmu_inverse_weights[:,None], tf.float32))
        
#         class_prototypes = tf.scatter_add(class_prototypes, bmu_idxs, bmus_update, name="implement_bmu_update")
    
#         with tf.control_dependencies([bmus_update]):
# #             class_prototypes = tf.scatter_add(class_prototypes, bmu_idxs, bmus_update, name="implement_bmu_update")
        
#             class_prototypes = tf.scatter_add(class_prototypes, bmu_idxs, data_minus_bmu, name="implement_bmu_update")
        
        
    return 0.5*loss1 + 0.5*loss2
    




