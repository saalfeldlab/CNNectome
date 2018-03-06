import tensorflow as tf


def tf_var_summary(var):
    # compute mean of variable
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean_'+var.name, mean)

    # compute std of variable
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))

    tf.summary.scalar('stddev_'+var.name, stddev)
    tf.summary.scalar('max_'+var.name, tf.reduce_max(var))
    tf.summary.scalar('min_'+var.name, tf.reduce_min(var))
    tf.summary.histogram('histogram_'+var.name, var)