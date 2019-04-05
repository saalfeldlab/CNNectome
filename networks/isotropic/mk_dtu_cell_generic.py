from networks import custom_fw_unet as unet
from networks import ops3d
import tensorflow as tf
import json
from utils.label import *
import logging

def make_net(labels, input_shape, loss_name='loss_total', mode='train'):
    names = dict()
    raw = tf.placeholder(tf.float32, shape=input_shape)
    names['raw'] = raw.name
    raw_bc = tf.reshape(raw, (1, 1,) + input_shape)

    last_fmap, fov, anisotropy = unet.unet(raw_bc, [12, 12*6, 12*6*6, 12*6*6*6], [48, 12*6, 12*6*6, 12*6*6*6],
                                           [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
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
    names['dist'] = dist_c.name
    network_outputs = tf.unstack(dist_c, len(labels), axis=0)
    if mode.lower() == 'train' or mode.lower() == 'training':
        # mask = tf.placeholder(tf.float32, shape=output_shape)
        # ribo_mask = tf.placeholder(tf.float32, shape=output_shape)

        gt = []
        w = []
        cw = []
        masks = []
        for l in labels:
            masks.append(tf.placeholder(tf.float32, shape=output_shape))
            gt.append(tf.placeholder(tf.float32, shape=output_shape))
            w.append(tf.placeholder(tf.float32, shape=output_shape))
            cw.append(l.class_weight)

        lb = []
        lub = []
        for output_it, gt_it, w_it, m_it, label in zip(network_outputs, gt, w, masks, labels):
            lb.append(tf.losses.mean_squared_error(gt_it, output_it, w_it * m_it))
            lub.append(tf.losses.mean_squared_error(gt_it, output_it, m_it))
            names[label.labelname] = output_it.name
            names['gt_'+label.labelname] = gt_it.name
            names['w_'+label.labelname] = w_it.name
            names['mask_'+ label.labelname] = m_it.name
        for label, lb_it, lub_it in zip(labels, lb, lub):
            tf.summary.scalar('lb_'+label.labelname, lb_it)
            tf.summary.scalar('lub_'+label.labelname, lub_it)
            names['lb_' + label.labelname] = lb_it.name
            names['lb_' + label.labelname] = lub_it.name

        loss_total = tf.add_n(lb)
        loss_total_unbalanced = tf.add_n(lub)
        loss_total_classweighted = tf.tensordot(lb, cw, axes=1)
        loss_total_unbalanced_classweighted = tf.tensordot(lub, cw, axes=1)

        tf.summary.scalar('loss_total', loss_total)
        names['loss_total'] = loss_total.name
        tf.summary.scalar('loss_total_unbalanced', loss_total_unbalanced)
        names['loss_total_unbalanced'] = loss_total_unbalanced.name
        tf.summary.scalar('loss_total_classweighted', loss_total_classweighted)
        names['loss_total_classweighted'] = loss_total_classweighted.name
        tf.summary.scalar('loss_total_unbalanced_classweighted', loss_total_unbalanced_classweighted)
        names['loss_total_unbalanced_classweighted'] = loss_total_unbalanced_classweighted.name

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
        names['optimizer'] = optimizer.name
        merged = tf.summary.merge_all()
        names['summary'] = merged.name

        with open('net_io_names.json', 'w') as f:
            json.dump(names, f)
    elif mode.lower() == 'inference' or mode.lower() == 'prediction' or mode.lower == 'pred':
        pass
    else:
        raise ValueError("unknown mode for network construction {0:}".format(mode))
    net_name = 'unet_'+mode
    tf.train.export_meta_graph(filename=net_name+'.meta')
    return net_name, output_shape

# def inference_net(labels):
#     input_shape = (340, 340, 340)
#     raw = tf.placeholder(tf.float32, shape=input_shape)
#     raw_bc = tf.reshape(raw, (1, 1,) + input_shape)
#
#     last_fmap, fov, anisotropy = unet.unet(raw_bc,
#                                            [12, 12 * 6, 12 * 6 * 6, 12 * 6 * 6 * 6],
#                                            [48, 12 * 6, 12 * 6 * 6, 12 * 6 * 6 * 6],
#                                            [[2, 2, 2], [2, 2, 2], [3, 3, 3]],
#                                            [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
#                                             [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
#                                            [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
#                                             [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
#                                            voxel_size=(1, 1, 1), fov=(1, 1, 1))
#
#     dist_bc, fov = ops3d.conv_pass(
#             last_fmap,
#             kernel_size=[[1, 1, 1]],
#             num_fmaps=len(labels),
#             activation=None,
#             fov=fov,
#             voxel_size=anisotropy
#             )
#
#     output_shape_bc = dist_bc.get_shape().as_list()
#     output_shape_c = output_shape_bc[1:]  # strip the batch dimension
#     output_shape = output_shape_c[1:]
#
#     dist_c = tf.reshape(dist_bc, output_shape_c)
#     network_outputs = tf.unstack(
#         dist_c, len(labels), axis=0)
#     tf.train.export_meta_graph(filename='unet_inference.meta')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    data_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v020719_o505x505x505_m1170x1170x1170/{0:}.n5"
    data_sources = list()
    data_sources.append(N5Dataset('crop1', 500*500*100, data_dir=data_dir))
    data_sources.append(N5Dataset('crop3', 400*400*250, data_dir=data_dir))
    data_sources.append(N5Dataset('crop4', 300*300*238, data_dir=data_dir))
    data_sources.append(N5Dataset('crop6', 250*250*250, special_categories=('ribosomes',), data_dir=data_dir))
    data_sources.append(N5Dataset('crop7', 300*300*80, special_categories=('ribosomes',), data_dir=data_dir))
    data_sources.append(N5Dataset('crop8', 200*200*100, data_dir=data_dir))
    data_sources.append(N5Dataset('crop9', 100*100*53, data_dir=data_dir))
    data_sources.append(N5Dataset('crop13', 160*160*110, special_categories=('ribosomes',), data_dir=data_dir))
    data_sources.append(N5Dataset('crop14', 150*150*65, data_dir=data_dir))
    data_sources.append(N5Dataset('crop15', 150*150*64, data_dir=data_dir))
    data_sources.append(N5Dataset('crop18', 200*200*110, data_dir=data_dir))
    data_sources.append(N5Dataset('crop19', 150*150*55, data_dir=data_dir))
    data_sources.append(N5Dataset('crop20', 200*200*85, data_dir=data_dir))
    data_sources.append(N5Dataset('crop21', 160*160*55, data_dir=data_dir))
    data_sources.append(N5Dataset('crop22', 170*170*100, data_dir=data_dir))

    ribo_sources = filter_by_category(data_sources, 'ribosomes')

    dt_scaling_factor = 50
    max_iteration = 500000
    loss_name = 'loss_total'
    labels = list()
    labels.append(Label('ecs', 1, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('plasma_membrane', 2, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('mito', (3, 4, 5), data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('mito_membrane', 3, scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('mito_DNA', 5, scale_loss=False, scale_key=labels[-2].scale_key, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('golgi', (6, 7), data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('golgi_membrane', 6, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('vesicle', (8, 9), data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('vesicle_membrane', 8, scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('MVB', (10, 11), data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('MVB_membrane', 10, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('lysosome', (12, 13), data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('lysosome_membrane', 12, scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('LD', (14, 15), data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('LD_membrane', 14, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('er', (16, 17, 18, 19, 20, 21, 22, 23), data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('er_membrane', (16, 18, 20), scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('ERES', (18, 19), data_sources=data_sources, data_dir=data_dir))
    #labels.append(Label('ERES_membrane', 18, scale_loss=False, scale_key=labels[-1].scale_key,
    #                    data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('nucleus', (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36), data_sources=data_sources,
                        data_dir=data_dir))
    labels.append(Label('nucleolus', 29, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('NE', (20, 21, 22, 23), scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources, data_dir=data_dir))
    #labels.append(Label('NE_membrane', (20, 22, 23), scale_loss=False, scale_key=labels[-1].scale_key,
    # data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('nuclear_pore', (22, 23), data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('nuclear_pore_out', 22, scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('chromatin', (24, 25, 26, 27, 36), data_sources=data_sources, data_dir=data_dir))
    #labels.append(Label('NHChrom', 25, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources,
    # data_dir=data_dir))
    #labels.append(Label('EChrom', 26, scale_loss=False, scale_key=labels[-2].scale_key, data_sources=data_sources,
    # data_dir=data_dir))
    #labels.append(Label('NEChrom', 27, scale_loss=False, scale_key=labels[-3].scale_key, data_sources=data_sources,
    # data_dir=data_dir))
    labels.append(Label('NHChrom', 25, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('EChrom', 26, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('NEChrom', 27, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('microtubules', (30, 31), data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('centrosome', (31, 32, 33), data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('distal_app', 32, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('subdistal_app', 33, data_sources=data_sources, data_dir=data_dir))
    labels.append(Label('ribosomes', 1, data_sources=ribo_sources, data_dir=data_dir))
    make_net(labels, (214,214,214), loss_name)
    #make_net(labels, (198,198,198), loss_name)
    #tf.reset_default_graph()
    #inference_net(labels)
