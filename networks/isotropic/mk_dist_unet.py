from networks import unet, ops3d
import tensorflow as tf
import json

if __name__ == "__main__":

    raw = tf.placeholder(tf.float32, shape=(196,)*3)
    raw_bc = tf.reshape(raw, (1, 1,) + (196,)*3)

    last_fmap, fov, anisotropy = unet.unet(raw_bc, 12, 6, [[2, 2, 2], [2, 2, 2], [3, 3, 3]],
                                           [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=(1, 1, 1), fov=(1, 1, 1))

    dist_bc, fov = ops3d.conv_pass(
            last_fmap,
            kernel_size=[[1, 1, 1]],
            num_fmaps=1,
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )
    output_shape_bc = dist_bc.get_shape().as_list()

    output_shape_c = output_shape_bc[1:] # strip the batch dimension
    output_shape = output_shape_c[1:] # strip the batch dimension

    dist = tf.reshape(dist_bc, output_shape)

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
    #    custom_ops.tf_var_summary(trainable)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename='build.meta')

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