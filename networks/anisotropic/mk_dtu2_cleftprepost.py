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
            num_fmaps=3,
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )

    cleft_dist, pre_dist, post_dist = tf.unstack(dist_batched, 3, axis=1)

    output_shape = cleft_dist.get_shape().as_list()

    gt_cleft_dist = tf.placeholder(tf.float32, shape=output_shape[1:])
    gt_pre_dist = tf.placeholder(tf.float32, shape=output_shape[1:])
    gt_post_dist = tf.placeholder(tf.float32, shape=output_shape[1:])
    gt_cleft_dist_batched = tf.reshape(gt_cleft_dist, shape=output_shape)
    gt_pre_dist_batched   = tf.reshape(gt_pre_dist,   shape=output_shape)
    gt_post_dist_batched  = tf.reshape(gt_post_dist,  shape=output_shape)


    loss_weights_cleft = tf.placeholder(tf.float32, shape=output_shape[1:])
    loss_weights_pre   = tf.placeholder(tf.float32, shape=output_shape[1:])
    loss_weights_post  = tf.placeholder(tf.float32, shape=output_shape[1:])
    loss_weights_cleft_batched = tf.reshape(loss_weights_cleft, shape=output_shape)
    loss_weights_pre_batched   = tf.reshape(loss_weights_pre,   shape=output_shape)
    loss_weights_post_batched  = tf.reshape(loss_weights_post,  shape=output_shape)

    loss_balanced_cleft = tf.losses.mean_squared_error(
        gt_cleft_dist_batched,
        cleft_dist,
        loss_weights_cleft_batched
    )
    loss_balanced_pre = tf.losses.mean_squared_error(
        gt_pre_dist_batched,
        pre_dist,
        loss_weights_pre_batched
    )
    loss_balanced_post = tf.losses.mean_squared_error(
        gt_post_dist_batched,
        post_dist,
        loss_weights_post_batched
    )
    loss_unbalanced_cleft = tf.losses.mean_squared_error(gt_cleft_dist_batched, cleft_dist)
    loss_unbalanced_pre = tf.losses.mean_squared_error(gt_pre_dist_batched, pre_dist)
    loss_unbalanced_post = tf.losses.mean_squared_error(gt_post_dist_batched, post_dist)

    loss_total = loss_balanced_cleft + loss_unbalanced_pre + loss_unbalanced_post
    tf.summary.scalar('loss_balanced_syn', loss_balanced_cleft)
    tf.summary.scalar('loss_balanced_pre', loss_balanced_pre)
    tf.summary.scalar('loss_balanced_post', loss_balanced_post)

    tf.summary.scalar('loss_unbalanced_syn', loss_unbalanced_cleft)
    tf.summary.scalar('loss_unbalanced_pre', loss_unbalanced_pre)
    tf.summary.scalar('loss_unbalanced_post', loss_unbalanced_post)
    tf.summary.scalar('loss_total', loss_total)

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
        'cleft_dist': cleft_dist.name,
        'pre_dist': pre_dist.name,
        'post_dist': post_dist.name,
        'gt_cleft_dist': gt_cleft_dist.name,
        'gt_pre_dist': gt_pre_dist.name,
        'gt_post_dist': gt_post_dist.name,
        'loss_balanced_cleft': loss_balanced_cleft.name,
        'loss_balanced_pre': loss_balanced_pre.name,
        'loss_balanced_post': loss_balanced_post.name,
        'loss_unbalanced_cleft': loss_unbalanced_cleft.name,
        'loss_unbalanced_pre': loss_unbalanced_pre.name,
        'loss_unbalanced_post': loss_unbalanced_post.name,
        'loss_total': loss_total.name,
        'loss_weights_cleft': loss_weights_cleft.name,
        'loss_weights_pre': loss_weights_pre.name,
        'loss_weights_post': loss_weights_post.name,
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
            num_fmaps=3,
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )

    cleft_dist, pre_dist, post_dist = tf.unstack(dist_batched, 3, axis=1)

    tf.train.export_meta_graph(filename='unet_inference.meta')


if __name__ == '__main__':
    train_net()
    tf.reset_default_graph()
    inference_net()
