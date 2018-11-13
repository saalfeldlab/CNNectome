from networks import unet, ops3d
import tensorflow as tf
import json


def train_net():
    input_shape = (43, 430, 430)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_bc = tf.reshape(raw, (1, 1,) + input_shape)

    last_fmap, fov, anisotropy = unet.unet(raw_bc, 12, 6, [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
                                           [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=(10, 1, 1), fov=(10, 1, 1))

    dist_bc, fov = ops3d.conv_pass(
            last_fmap,
            kernel_size=[[1, 1, 1]],
            num_fmaps=2,
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )

    output_shape_bc = dist_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]
    output_shape = output_shape_c[1:]
    dist_c = tf.reshape(dist_bc, output_shape_c)
    syn_dist, bdy_dist = tf.unstack(dist_c, 2, axis=0)

    gt_syn_dist = tf.placeholder(tf.float32, shape=output_shape)
    gt_bdy_dist = tf.placeholder(tf.float32, shape=output_shape)

    loss_weights = tf.placeholder(tf.float32, shape=output_shape)
    mask = tf.placeholder(tf.float32, shape=output_shape)

    loss_balanced_syn = tf.losses.mean_squared_error(
        gt_syn_dist,
        syn_dist,
        loss_weights
    )
    loss_bdy = tf.losses.mean_squared_error(gt_bdy_dist, bdy_dist)
    loss_total = loss_balanced_syn + loss_bdy
    tf.summary.scalar('loss_balanced_syn', loss_balanced_syn)
    tf.summary.scalar('loss_bdy', loss_bdy)
    tf.summary.scalar('loss_total', loss_total)

    loss_unbalanced = tf.losses.mean_squared_error(gt_syn_dist, syn_dist, mask)
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
        'mask': mask.name,
        'optimizer': optimizer.name,
        'summary': merged.name}

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)


def inference_net():
    input_shape = (91, 862, 862)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_bc = tf.reshape(raw, (1, 1,) + input_shape)

    last_fmap, fov, anisotropy = unet.unet(raw_bc, 12, 6, [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
                                           [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=(10, 1, 1), fov=(10, 1, 1))

    dist_bc, fov = ops3d.conv_pass(
            last_fmap,
            kernel_size=[[1, 1, 1]],
            num_fmaps=2,
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )

    output_shape_bc = dist_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]
    #output_shape = output_shape_c[1:]
    dist_c = tf.reshape(dist_bc, output_shape_c)
    syn_dist, bdy_dist = tf.unstack(dist_c, 2, axis=0)

    tf.train.export_meta_graph(filename='unet_inference.meta')


if __name__ == '__main__':
    train_net()
    tf.reset_default_graph()
    inference_net()
