from networks import unet
import tensorflow as tf
import json


def train_net():
    input_shape = (43, 430, 430)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1, 1,) + input_shape)

    last_fmap, fov, anisotropy = unet.unet(raw_batched, 12, [3, 3, 6], [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
                                           [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=(10, 1, 1), fov=(10, 1, 1))

    dist_batched, fov = unet.conv_pass(
            last_fmap,
            kernel_size=[[1, 1, 1]],
            num_fmaps=1,
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )

    output_shape_batched = dist_batched.get_shape().as_list()

    output_shape = output_shape_batched[1:]  # strip the batch dimension

    dist = tf.reshape(dist_batched, output_shape)

    gt_dist = tf.placeholder(tf.float32, shape=output_shape)

    loss_weights = tf.placeholder(tf.float32, shape=output_shape[1:])
    loss_weights_batched = tf.reshape(loss_weights, shape=output_shape)

    loss_balanced = tf.losses.mean_squared_error(
        gt_dist,
        dist,
        loss_weights_batched
    )
    tf.summary.scalar('loss_balanced_syn', loss_balanced)

    loss_unbalanced = tf.losses.mean_squared_error(gt_dist, dist)
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
        'dist': dist.name,
        'gt_dist': gt_dist.name,
        'loss_balanced_syn': loss_balanced.name,
        'loss_unbalanced_syn': loss_unbalanced.name,
        'loss_weights': loss_weights.name,
        'optimizer': optimizer.name,
        'summary': merged.name}

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)


def inference_net():
    input_shape = (91, 862, 862)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1, 1,) + input_shape)

    last_fmap, fov, anisotropy = unet.unet(raw_batched, 12, 6, [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
                                           [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=(10, 1, 1), fov=(10, 1, 1))

    dist_batched, fov = unet.conv_pass(
        last_fmap,
        kernel_size=[[1, 1, 1]],
        num_fmaps=1,
        activation=None,
        fov=fov,
        voxel_size=anisotropy
    )

    output_shape_batched = dist_batched.get_shape().as_list()

    output_shape = output_shape_batched[1:]

    dist = tf.reshape(dist_batched, output_shape)

    tf.train.export_meta_graph(filename='unet_inference.meta')


if __name__ == '__main__':
    train_net()
    tf.reset_default_graph()
    inference_net()
