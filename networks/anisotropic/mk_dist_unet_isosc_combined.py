import networks
import tensorflow as tf
import json

if __name__ == "__main__":
    print(networks.__file__)
    raw = tf.placeholder(tf.float32, shape=(43, 430, 430))
    raw_batched = tf.reshape(raw, (1, 1,) + (43, 430, 430))

    #unet = networks.unet(raw_batched, 12, 6, [[1, 3, 3], [1, 3, 3], [3, 3, 3]], anisotropy=10)
    unet, fov, anisotropy = networks.unet(raw_batched, 12, 6,
                                          [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
                                          [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3),(1, 3, 3)],[(3, 3, 3),(3, 3, 3)],
                                           [(3, 3, 3), (3, 3, 3)]],
                                          [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                           [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=[10, 1, 1], fov=[10, 1, 1])
    # raw = tf.placeholder(tf.float32, shape=(132,)*3)
    # raw_batched = tf.reshape(raw, (1, 1,) + (132,)*3)
    #
    # unet = networks.unet(raw_batched, 24, 3, [[2, 2, 2], [2, 2, 2], [2, 2, 2]])

    dist_batched, fov1 = networks.conv_pass(
        unet,
        kernel_size=[[1,1,1]],
        num_fmaps=1,
        activation=None,
        fov=fov,
        voxel_size=anisotropy
        )
    aff_batched, fov = networks.conv_pass(
        unet,
        kernel_size=[[1,1,1]],
        num_fmaps=3,
        activation=None,
        name='aff_conv',
        fov=fov,
        voxel_size=anisotropy
        )
    print("distbatched",dist_batched.get_shape().as_list())
    dist_output_shape_batched = dist_batched.get_shape().as_list()
    dist_output_shape = dist_output_shape_batched[1:]
    syn_dist = tf.reshape(dist_batched, dist_output_shape)
    gt_syn_dist = tf.placeholder(tf.float32, shape=dist_output_shape)
    loss_weights = tf.placeholder(tf.float32, shape=dist_output_shape[1:])
    loss_weights_batched = tf.reshape(loss_weights, shape=dist_output_shape)

    aff_output_shape_batched = aff_batched.get_shape().as_list()
    print("aff_batched", aff_output_shape_batched)
    aff_output_shape = aff_output_shape_batched[1:]
    print("aff_output", aff_output_shape)
    aff = tf.reshape(aff_batched, aff_output_shape)

    gt_aff = tf.placeholder(tf.float32, shape=aff_output_shape)
    #syn_dist, bdy_dist = tf.unstack(dist_batched, 2, axis=1)
    #print("after unstack", syn_dist.get_shape().as_list(), bdy_dist.get_shape().as_list())
    #output_shape = syn_dist.get_shape().as_list()

    #output_shape = output_shape_batched[1:] # strip the batch dimension

    #gt_bdy_dist = tf.placeholder(tf.float32, shape=output_shape)


    syn_loss_eucl = tf.losses.mean_squared_error(
        gt_syn_dist,
        syn_dist,
        loss_weights_batched
    )
    bdy_loss_eucl = tf.losses.mean_squared_error(gt_aff, aff)#tf.losses.mean_squared_error(gt_bdy_dist, bdy_dist)
    loss_combined = bdy_loss_eucl + syn_loss_eucl

    tf.summary.scalar('loss_bdy', bdy_loss_eucl)
    tf.summary.scalar('loss_total', syn_loss_eucl)
    tf.summary.scalar('loss_bdy_plus_syn', loss_combined)

   # mae = tf.losses.absolute_difference(gt_dist, dist)
    #tf.summary.scalar('loss_mae', mae)

    mse = tf.losses.mean_squared_error(gt_syn_dist, syn_dist)
    tf.summary.scalar('loss_mse_unbalanced', mse)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)

    optimizer = opt.minimize(loss_combined)
    #for trainable in tf.trainable_variables():
    #    networks.tf_var_summary(trainable)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename='unet.meta')

    names = {
        'raw': raw.name,
        'syn_dist': syn_dist.name,
        'gt_syn_dist': gt_syn_dist.name,
        'bdy_dist': aff.name,
        'gt_bdy_dist': gt_aff.name,
        'loss_combined': loss_combined.name,
        'loss_syn': syn_loss_eucl.name,
        'loss_bdy': bdy_loss_eucl.name,
        'loss_weights': loss_weights.name,
        'optimizer': optimizer.name,
        'summary': merged.name
    }

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)