import numpy as np
import tensorflow as tf



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


def get_rmse_loss(modality1_dist, modality2_dist, flags):
    lower_tril_idxs = np.column_stack(np.tril_indices(n=flags.train_batch_size, k=-1))
    x = tf.expand_dims(tf.gather_nd(modality1_dist, lower_tril_idxs), axis=-1) 
    y = tf.expand_dims(tf.gather_nd(modality2_dist, lower_tril_idxs), axis=-1)
#     x_max = tf.reduce_max(tf.stop_gradient(x))
#     y_max = tf.reduce_max(tf.stop_gradient(y))
#     loss = tf.norm(tf.exp(x-x_max)-tf.exp(y-y_max)) / len(lower_tril_idxs)
#     x = tf.stop_gradient(x)
    y_const = tf.stop_gradient(y)
    x_const = tf.stop_gradient(x)
    loss1 = tf.div_no_nan( tf.square(tf.norm(x - y_const)), tf.norm(x) * tf.norm(y_const) )
    loss2 = tf.div_no_nan( tf.square(tf.norm(x_const - y)), tf.norm(x_const) * tf.norm(y) )
    return 0.5*loss1 + 0.5*loss2

