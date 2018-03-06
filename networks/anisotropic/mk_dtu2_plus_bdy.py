from networks import unet
import tensorflow as tf
import json


def train_net():
    input_shape = (43, 430, 430)
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
            num_fmaps=2,
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )

    syn_dist, bdy_dist = tf.unstack(dist_batched, 2, axis=1)

    output_shape = syn_dist.get_shape().as_list()

    gt_syn_dist = tf.placeholder(tf.float32, shape=output_shape)
    gt_bdy_dist = tf.placeholder(tf.float32, shape=output_shape)

    loss_weights = tf.placeholder(tf.float32, shape=output_shape[1:])
    loss_weights_batched = tf.reshape(loss_weights, shape=output_shape)

    loss_balanced_syn = tf.losses.mean_squared_error(
        gt_syn_dist,
        syn_dist,
        loss_weights_batched
    )
    loss_bdy = tf.losses.mean_squared_error(gt_bdy_dist, bdy_dist)
    loss_total = loss_balanced_syn + loss_bdy
    tf.summary.scalar('loss_balanced_syn', loss_balanced_syn)
    tf.summary.scalar('loss_bdy', loss_bdy)
    tf.summary.scalar('loss_total', loss_total)

    loss_unbalanced = tf.losses.mean_squared_error(gt_syn_dist, syn_dist)
    tf.summary.scalar('loss_unbalanced_syn', loss_unbalanced)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)

    optimizer = opt.minimize(loss_total)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename='unet.meta')

    names = {
        'raw': raw.name,
        'syn_dist': syn_dist.name,
        'bdy_dist': bdy_dist.name,
        'gt_syn_dist': gt_syn_dist.name,
        'gt_bdy_dist': gt_bdy_dist.name,
        'loss_balanced_syn': loss_balanced_syn.name,
        'loss_unbalanced_syn': loss_unbalanced.name,
        'loss_bdy': loss_bdy.name,
        'loss_total': loss_total.name,
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
            num_fmaps=2,
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )

    syn_dist, bdy_dist = tf.unstack(dist_batched, 2, axis=1)

    output_shape_batched = dist_batched.get_shape().as_list()

    output_shape = output_shape_batched[1:]

    tf.train.export_meta_graph(filename='unet_inference.meta')


if __name__ == '__main__':
    train_net()
    tf.reset_default_graph()
    inference_net()
