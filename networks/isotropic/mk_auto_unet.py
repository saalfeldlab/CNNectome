from networks import unet_auto, ops3d
import tensorflow as tf
import json

if __name__ == "__main__":

    raw = tf.placeholder(tf.float32, shape=(132,)*3)
    raw_batched = tf.reshape(raw, (1, 1,) + (132,)*3)

    pred = tf.placeholder(tf.float32, shape=(132,)*3)
    pred_batched = tf.reshape(pred, (1,1,)+(132,)*3)

    last_fmap, fov, anisotropy = unet_auto.unet_auto(raw_batched, pred_batched, 24, 3,
                                                     [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                                                     [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                                      [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                                     [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                                      [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3,3)]],
                                                     [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                                      [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]]
                                                     )

    affs_batched, fov = ops3d.conv_pass(
        last_fmap,
        kernel_size=[[1,1,1]],
        num_fmaps=3,
        activation='sigmoid',
        fov=fov,
        voxel_size=anisotropy
    )

    output_shape_batched = affs_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:] # strip the batch dimension

    affs = tf.reshape(affs_batched, output_shape)

    gt_affs = tf.placeholder(tf.float32, shape=output_shape)

    loss_weights = tf.placeholder(tf.float32, shape=output_shape)

    loss = tf.losses.mean_squared_error(
        gt_affs,
        affs,
        loss_weights)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    tf.train.export_meta_graph(filename='unet_auto.meta')

    names = {
        'raw': raw.name,
        'affs': affs.name,
        'gt_affs': gt_affs.name,
        'loss_weights': loss_weights.name,
        'loss': loss.name,
        'optimizer': optimizer.name}
    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)