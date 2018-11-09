from networks import custom_fw_unet as unet
from networks import ops3d
import tensorflow as tf
import json
from training.isotropic.train_cell2 import Label, compute_total_voxels
import logging

def train_net(labels):
    input_shape = (196, 196, 196)
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
    network_outputs = tf.unstack(dist_c, len(labels), axis=0)

    gt = []
    w = []
    cw = []
    for l in labels:
        gt.append(tf.placeholder(tf.float32, shape=output_shape))
        w.append(tf.placeholder(tf.float32, shape=output_shape))
        cw.append(l.class_weight)

    lb = []
    lub = []
    for output_it, gt_it, w_it in zip(network_outputs, gt, w):
        lb.append(tf.losses.mean_squared_error(gt_it, output_it, w_it))
        lub.append(tf.losses.mean_squared_error(gt_it, output_it))
    for label, lb_it, lub_it in zip(labels, lb, lub):
        tf.summary.scalar('lb_'+label.labelname, lb_it)
        tf.summary.scalar('lub_'+label.labelname, lub_it)


    loss_total = tf.add_n(lb)
    loss_total_unbalanced = tf.add_n(lub)
    loss_total_classweighted = tf.tensordot(lb, cw, axes=1)
    loss_total_unbalanced_classweighted = tf.tensordot(lub, cw, axes=1)

    tf.summary.scalar('loss_total', loss_total)
    tf.summary.scalar('loss_total_unbalanced', loss_total_unbalanced)
    tf.summary.scalar('loss_total_classweighted', loss_total_classweighted)
    tf.summary.scalar('loss_total_unbalanced_classweighted', loss_total_unbalanced_classweighted)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    if loss_name == 'loss_total':
        optimizer = opt.minimize(loss_total)
    elif loss_name == 'loss_total_unbalanced':
        optimizer = opt.minimize(loss_total_unbalanced)
    elif loss_name == 'loss_total_unbalanced_classweighted':
        optimizer = opt.minimize(loss_total_unbalanced_classweighted)
    elif loss_name == 'loss_total_classweighted':
        optimizer = opt.minimize(loss_total_classweighted)
    else:
        raise ValueError(loss_name + " not defined")
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename='unet.meta')

    names = {
        'raw': raw.name,
        'dist': dist_c.name,
        'loss_total': loss_total.name,
        'loss_total_unbalanced': loss_total_unbalanced.name,
        'loss_total_classweighted': loss_total_classweighted.name,
        'loss_total_unbalanced_classweighted': loss_total_unbalanced_classweighted.name,
        'optimizer': optimizer.name,
        'summary': merged.name
    }

    for label, output_it, gt_it, w_it, lb_it, lub_it in zip(labels, network_outputs, gt, w, lb, lub):
        names[label.labelname] = output_it.name
        names['gt_' + label.labelname] = gt_it.name
        names['w_' + label.labelname] = w_it.name
        names['lb_' + label.labelname] = lb_it.name
        names['lub_' + label.labelname] = lub_it.name

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)


def inference_net(labels):
    input_shape = (340, 340, 340)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_bc = tf.reshape(raw, (1, 1,) + input_shape)

    last_fmap, fov, anisotropy = unet.unet(raw_bc,
                                           [12, 12 * 6, 12 * 6 * 6, 12 * 6 * 6 * 6],
                                           [48, 12 * 6, 12 * 6 * 6, 12 * 6 * 6 * 6],
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
    logging.basicConfig(level=logging.INFO)
    data_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cell/{0:}.n5"
    data_sources = ['hela_cell2_crop1_110618', 'hela_cell2_crop8_110618', 'hela_cell2_crop9_110618',
                    'hela_cell2_crop14_110618', 'hela_cell2_crop15_110618']
    input_shape = (196, 196, 196)
    output_shape = (92, 92, 92)
    dt_scaling_factor = 50
    max_iteration = 500000
    loss_name = 'loss_total'

    labels = []
    labels.append(Label('cell', (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), data_sources=data_sources))
    labels.append(Label('plasma_membrane', 2, data_sources=data_sources))
    labels.append(Label('ERES', (6, 7), data_sources=data_sources))
    labels.append(Label('ERES_membrane', 6, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources))
    labels.append(Label('MVB', (10, 11), data_sources=data_sources))
    labels.append(Label('MVB_membrane', 10, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources))
    labels.append(Label('er', (4, 5, 6, 7), data_sources=data_sources))
    labels.append(Label('er_membrane', (4, 6), scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources))
    labels.append(Label('mito', (8, 9), data_sources=data_sources))
    labels.append(Label('mito_membrane', 8, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources))
    labels.append(Label('vesicles', (12, 13), data_sources=data_sources))
    labels.append(Label('vesicles_membrane', 12, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources))
    labels.append(Label('microtubules', 14, data_sources=data_sources))

    train_net(labels, loss_name)
    tf.reset_default_graph()
    inference_net(labels)
