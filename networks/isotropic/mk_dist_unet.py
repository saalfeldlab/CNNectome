import networks
import tensorflow as tf
import json

if __name__ == "__main__":

    raw = tf.placeholder(tf.float32, shape=(196,)*3)
    raw_batched = tf.reshape(raw, (1, 1,) + (196,)*3)

    unet = networks.unet(raw_batched, 12, 6, [[2, 2, 2], [2, 2, 2], [3, 3, 3]])

    # raw = tf.placeholder(tf.float32, shape=(132,)*3)
    # raw_batched = tf.reshape(raw, (1, 1,) + (132,)*3)
    #
    # unet = networks.unet(raw_batched, 24, 3, [[2, 2, 2], [2, 2, 2], [2, 2, 2]])

    dist_batched = networks.conv_pass(
        unet,
        kernel_size=1,
        num_fmaps=1,
        num_repetitions=1,
        activation='tanh')

    output_shape_batched = dist_batched.get_shape().as_list()

    output_shape = output_shape_batched[1:] # strip the batch dimension

    dist = tf.reshape(dist_batched, output_shape)

    gt_dist = tf.placeholder(tf.float32, shape=output_shape)

    #loss_weights = tf.placeholder(tf.float32, shape=output_shape)

    loss_eucl = tf.losses.mean_squared_error(
        gt_dist,
        dist)
    tf.summary.scalar('loss_total', loss_eucl)

   # mae = tf.losses.absolute_difference(gt_dist, dist)
    #tf.summary.scalar('loss_mae', mae)

    #mse = tf.losses.mean_squared_error(gt_dist, dist)
    #tf.summary.scalar('loss_mse_unbalanced', mse)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)

    optimizer = opt.minimize(loss_eucl)
    #for trainable in tf.trainable_variables():
    #    networks.tf_var_summary(trainable)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename='unet.meta')

    names = {
        'raw': raw.name,
        'dist': dist.name,
        'gt_dist': gt_dist.name,
        'loss': loss_eucl.name,
        #'loss_weights': loss_weights.name,
        'optimizer': optimizer.name,
        'summary': merged.name}

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)