import tensorflow as tf
import tensorflow.contrib.slim as slim


def get_classifier_loss(features_modality1, features_modality2, labels, flags, scope="classifier_loss"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        labels = tf.concat([labels, labels], axis=0)
        
        logits1 = slim.fully_connected(features_modality1, num_outputs=flags.num_classes_train, activation_fn=None, 
                                       normalizer_fn=None, trainable=True, weights_regularizer=None, 
                                       scope='classifier_fc1')
        logits2 = slim.fully_connected(features_modality2, num_outputs=flags.num_classes_train, activation_fn=None, 
                                       normalizer_fn=None, trainable=True, weights_regularizer=None, 
                                       scope='classifier_fc2')
        logits = tf.concat([logits1, logits2], axis=0)
        
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        
        accuracy = slim.metrics.accuracy(tf.argmax(logits, -1), labels)
        
        tf.summary.scalar('metrics/loss', loss)
        tf.summary.scalar('metrics/accuracy', accuracy)
        
    return loss

