from networks import unet, ops3d
import tensorflow as tf
import json


def train_net():
    input_shape = (196, 196, 196)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_bc = tf.reshape(raw, (1, 1,) + input_shape)

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

    output_shape_c = output_shape_bc[1:]  # strip the batch dimension

    dist_c = tf.reshape(dist_bc, output_shape_c)

    gt_dist = tf.placeholder(tf.float32, shape=output_shape_c[1:])
    gt_dist_c = tf.reshape(gt_dist, shape=output_shape_c)

    loss_weights = tf.placeholder(tf.float32, shape=output_shape_c[1:])
    loss_weights_c = tf.reshape(loss_weights, shape=output_shape_c)

    loss_balanced = tf.losses.mean_squared_error(
        gt_dist_c,
        dist_c,
        loss_weights_c
    )
    tf.summary.scalar('loss_balanced_syn', loss_balanced)

    loss_unbalanced = tf.losses.mean_squared_error(gt_dist_c, dist_c)
    tf.summary.scalar('loss_unbalanced_syn', loss_unbalanced)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)

    optimizer = opt.minimize(loss_balanced)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename='unet.meta')

    names = {
        'raw': raw.name,
        'dist': dist_c.name,
        'gt_dist': gt_dist.name,
        'loss_balanced_syn': loss_balanced.name,
        'loss_unbalanced_syn': loss_unbalanced.name,
        'loss_weights': loss_weights.name,
        'optimizer': optimizer.name,
        'summary': merged.name}

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)


def inference_net():
    input_shape = (196, 196, 196)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_bc = tf.reshape(raw, (1, 1,) + input_shape)

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

    output_shape_c = output_shape_bc[1:]

    dist_c = tf.reshape(dist_bc, output_shape_c)

    tf.train.export_meta_graph(filename='unet_inference.meta')


if __name__ == '__main__':
    train_net()
    tf.reset_default_graph()
    inference_net()