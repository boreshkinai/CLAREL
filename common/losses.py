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
    
    mi_loss = -hxy + hx + hy
    return -mi_loss