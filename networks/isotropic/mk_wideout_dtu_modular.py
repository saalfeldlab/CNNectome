from networks import custom_fw_unet as unet
from networks import ops3d
import tensorflow as tf
import json


def train_net(labels):
    input_shape = (196, 196, 196)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_bc = tf.reshape(raw, (1, 1,) + input_shape)

    last_fmap, fov, anisotropy = unet.unet(raw_bc, [12, 12*6, 12*6*6, 12*6*6*6], [48, 12*6, 12*6*6, 12*6*6*6], [[2, 2, 2], [2, 2, 2], [3, 3, 3]],
                                           [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=(1, 1, 1), fov=(1, 1, 1))

    dist_bc, fov = ops3d.conv_pass(
            last_fmap,
            kernel_size=[[1, 1, 1]],
            num_fmaps=len(labels),
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )
    output_shape_bc = dist_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]  # strip the batch dimension
    output_shape = output_shape_c[1:]

    dist_c = tf.reshape(dist_bc, output_shape_c)
    network_outputs = tf.unstack(
        dist_c,len(labels), axis=0)
    mask = tf.placeholder(tf.float32, shape=output_shape)

    gt = []
    w = []
    for l in range(len(labels)):
        gt.append(tf.placeholder(tf.float32, shape=output_shape))
        w.append(tf.placeholder(tf.float32, shape=output_shape))
    lb = []
    lub = []
    for output_it, gt_it, w_it in zip(network_outputs, gt, w):
        lb.append(tf.losses.mean_squared_error(gt_it, output_it, w_it))
        lub.append(tf.losses.mean_squared_error(gt_it, output_it, mask))
    for label, lb_it, lub_it in zip(labels, lb, lub):
        tf.summary.scalar('lb_'+label, lb_it)
        tf.summary.scalar('lub_'+label, lub_it)
    loss_total = tf.add_n(lb)
    loss_total_unbalanced = tf.add_n(lub)
    tf.summary.scalar('loss_total', loss_total)
    tf.summary.scalar('loss_total_unbalanced', loss_total_unbalanced)
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
        'dist': dist_c.name,
        'loss_total': loss_total.name,
        'loss_total_unbalanced': loss_total_unbalanced.name,
        'mask': mask.name,
        'optimizer': optimizer.name,
        'summary': merged.name
    }
    for label, output_it, gt_it, w_it, lb_it, lub_it in zip(labels, network_outputs, gt, w, lb, lub):
        names[label] = output_it.name
        names['gt_'+label] = gt_it.name
        names['w_'+label] = w_it.name
        names['lb_'+label] = lb_it.name
        names['lub_'+label] = lub_it.name

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)


def inference_net(labels):
    input_shape = (340, 340, 340)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_bc = tf.reshape(raw, (1, 1,) + input_shape)

    last_fmap, fov, anisotropy = unet.unet(raw_bc, [12, 12*6, 12*6*6, 12*6*6*6], [48, 12*6, 12*6*6, 12*6*6*6],
                                           [[2, 2, 2], [2, 2, 2], [3, 3, 3]],
                                           [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=(1, 1, 1), fov=(1, 1, 1))

    dist_bc, fov = ops3d.conv_pass(
            last_fmap,
            kernel_size=[[1, 1, 1]],
            num_fmaps=len(labels),
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )
    output_shape_bc = dist_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]  # strip the batch dimension
    output_shape = output_shape_c[1:]

    dist_c = tf.reshape(dist_bc, output_shape_c)
    network_outputs = tf.unstack(
        dist_c, len(labels), axis=0)
    tf.train.export_meta_graph(filename='unet_inference.meta')


if __name__ == '__main__':
    labels=['ECS', 'cell', 'plasma_membrane', 'ERES', 'ERES_membrane', 'MVB', 'MVB_membrane', 'er', 'er_membrane',
            'mito', 'mito_membrane', 'vesicles', 'microtubules']
    train_net(labels)
    tf.reset_default_graph()
    inference_net(labels)