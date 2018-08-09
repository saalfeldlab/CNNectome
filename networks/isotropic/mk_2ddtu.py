from networks import unet2d, ops2d
import tensorflow as tf
import json


def train_net():
    input_shape = (256, 256)
    raw_c = tf.placeholder(tf.float32, shape=(3,)+input_shape)
    raw_bc = tf.reshape(raw_c, (1, 3,) + input_shape)

    last_fmap, fov, anisotropy = unet2d.unet(raw_bc, 12, 6, [[3, 3], [3, 3], [3, 3]],
                                           [[(3, 3), (3, 3)], [(3, 3), (3, 3)],
                                            [(3, 3), (3, 3)], [(3, 3), (3, 3)]],
                                           [[(3, 3), (3, 3)], [(3, 3), (3, 3)],
                                            [(3, 3), (3, 3)], [(3, 3), (3, 3)]],
                                           voxel_size=(1, 1), fov=(1, 1))

    res_bc, fov = ops2d.conv_pass(
            last_fmap,
            kernel_size=[[1, 1]],
            num_fmaps=1,
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )
    output_shape_bc = res_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]
    output_shape = output_shape_c[1:]
    # res_c = tf.reshape(res_bc, output_shape_c)
    # aff1_0, aff1_1, aff3_0, aff3_1, aff9_0, aff9_1, dist = tf.unstack(res_c, 7, axis=0)
    # aff1_c = tf.stack([aff1_0, aff1_1], axis=0)
    # aff3_c = tf.stack([aff3_0, aff3_1], axis=0)
    # aff9_c = tf.stack([aff9_0, aff9_1], axis=0)
    dist = tf.reshape(res_bc, output_shape)
    gt_dist = tf.placeholder(tf.float32, shape=output_shape)
    # gt_aff_c = tf.placeholder(tf.float32, shape=[6, ]+output_shape)
    # gt_aff1_0, gt_aff1_1, gt_aff3_0, gt_aff3_1, gt_aff9_0, gt_aff9_1 = tf.unstack(gt_aff_c, 6, axis=0)
    # gt_aff1_c = tf.stack([gt_aff1_0, gt_aff1_1], axis=0)
    # gt_aff3_c = tf.stack([gt_aff3_0, gt_aff3_1], axis=0)
    # gt_aff9_c = tf.stack([gt_aff9_0, gt_aff9_1], axis=0)

    loss_weights_dist = tf.placeholder(tf.float32, shape=output_shape)
    # loss_weights_aff_c = tf.placeholder(tf.float32, shape=[2,]+output_shape)

    loss_dist = tf.losses.mean_squared_error(
        gt_dist,
        dist,
        loss_weights_dist
    )
    # loss_aff1 = tf.losses.mean_squared_error(gt_aff1_c, aff1_c, loss_weights_aff_c)
    # loss_aff3 = tf.losses.mean_squared_error(gt_aff3_c, aff3_c, loss_weights_aff_c)
    # loss_aff9 = tf.losses.mean_squared_error(gt_aff9_c, aff9_c, loss_weights_aff_c)
    # loss_total = loss_dist * 3 + loss_aff1 + loss_aff3 + loss_aff9
    tf.summary.scalar('loss_dist', loss_dist)
    # tf.summary.scalar('loss_aff1', loss_aff1)
    # tf.summary.scalar('loss_aff3', loss_aff3)
    # tf.summary.scalar('loss_aff9', loss_aff9)
    # tf.summary.scalar('loss_total', loss_total)

    loss_unbalanced_dist = tf.losses.mean_squared_error(gt_dist, dist)
    # loss_unbalanced_aff1 = tf.losses.mean_squared_error(gt_aff1_c, aff1_c)
    # loss_unbalanced_aff3 = tf.losses.mean_squared_error(gt_aff3_c, aff3_c)
    # loss_unbalanced_aff9 = tf.losses.mean_squared_error(gt_aff9_c, aff9_c)
    tf.summary.scalar('loss_unbalanced_dist', loss_unbalanced_dist)
    # tf.summary.scalar('loss_unbalanced_aff1', loss_unbalanced_aff1)
    # tf.summary.scalar('loss_unbalanced_aff3', loss_unbalanced_aff3)
    # tf.summary.scalar('loss_unbalanced_aff9', loss_unbalanced_aff9)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)

    optimizer = opt.minimize(loss_dist)
    # optimizer = opt.minimize(loss_total)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename='unet.meta')

    names = {
        'raw': raw_c.name,
        'dist': dist.name,
        # 'aff1': aff1_c.name,
        # 'aff3': aff3_c.name,
        # 'aff9': aff9_c.name,
        'gt_dist': gt_dist.name,
        # 'gt_aff': gt_aff_c.name,
        'loss_dist': loss_dist.name,
        # 'loss_aff1': loss_aff1.name,
        # 'loss_aff3': loss_aff3.name,
        # 'loss_aff9': loss_aff9.name,
        # 'loss_total': loss_total.name,
        'loss_unbalanced_dist': loss_unbalanced_dist.name,
        # 'loss_unbalanced_aff1': loss_unbalanced_aff1.name,
        # 'loss_unbalanced_aff3': loss_unbalanced_aff3.name,
        # 'loss_unbalanced_aff9': loss_unbalanced_aff9.name,
        'loss_weights_dist': loss_weights_dist.name,
        # 'loss_weights_aff': loss_weights_aff_c.name,
        'optimizer': optimizer.name,
        'summary': merged.name}

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)


def inference_net():
    input_shape = (256, 256)
    raw_c = tf.placeholder(tf.float32, shape=(3,)+input_shape)
    raw_bc = tf.reshape(raw_c, (1, 3,) + input_shape)
    last_fmap, fov, anisotropy = unet2d.unet(raw_bc, 12, 6, [[3, 3], [3, 3], [3, 3]],
                                             [[(3, 3), (3, 3)], [(3, 3), (3, 3)],
                                              [(3, 3), (3, 3)], [(3, 3), (3, 3)]],
                                             [[(3, 3), (3, 3)], [(3, 3), (3, 3)],
                                              [(3, 3), (3, 3)], [(3, 3), (3, 3)]],
                                             voxel_size=(1, 1), fov=(1, 1))

    res_bc, fov = ops2d.conv_pass(
        last_fmap,
        kernel_size=[[1, 1]],
        num_fmaps=1,
        activation=None,
        fov=fov,
        voxel_size=anisotropy
    )
    output_shape_bc = res_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]
    output_shape = output_shape_c[1:]
    # res_c = tf.reshape(res_bc, output_shape_c)
    # aff1_0, aff1_1, aff3_0, aff3_1, aff9_0, aff9_1, dist = tf.unstack(res_c, 7, axis=0)
    # aff1_c = tf.stack([aff1_0, aff1_1], axis=0)
    # aff3_c = tf.stack([aff3_0, aff3_1], axis=0)
    # aff9_c = tf.stack([aff9_0, aff9_1], axis=0)
    dist = tf.reshape(res_bc, output_shape)
    tf.train.export_meta_graph(filename='unet_inference.meta')


if __name__ == '__main__':
    train_net()
    tf.reset_default_graph()
    inference_net()
