from networks import unet, ops3d
import tensorflow as tf
import json


def train_net():
    input_shape = (196, 196, 196)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_bc = tf.reshape(raw, (1, 1,) + input_shape)

    last_fmap, fov, anisotropy = unet.unet(raw_bc, 12, 6, [[2, 2, 2], [2, 2, 2], [3, 3, 3]],
                                           [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=(1, 1, 1), fov=(1, 1, 1))

    dist_bc, fov = ops3d.conv_pass(
            last_fmap,
            kernel_size=[[1, 1, 1]],
            num_fmaps=11,
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )
    output_shape_bc = dist_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]  # strip the batch dimension
    output_shape = output_shape_c[1:]

    dist_c = tf.reshape(dist_bc, output_shape_c)
    centrosome, golgi, golgi_mem, er, er_mem, mvb, mvb_mem, mito, mito_mem, lysosome, lysosome_mem = tf.unstack(
        dist_c, 11, axis=0)

    gt_centrosome = tf.placeholder(tf.float32, shape=output_shape)
    gt_golgi = tf.placeholder(tf.float32, shape=output_shape)
    gt_golgi_mem = tf.placeholder(tf.float32, shape=output_shape)
    gt_er = tf.placeholder(tf.float32, shape=output_shape)
    gt_er_mem = tf.placeholder(tf.float32, shape=output_shape)
    gt_mvb = tf.placeholder(tf.float32, shape=output_shape)
    gt_mvb_mem = tf.placeholder(tf.float32, shape=output_shape)
    gt_mito = tf.placeholder(tf.float32, shape=output_shape)
    gt_mito_mem = tf.placeholder(tf.float32, shape=output_shape)
    gt_lysosome = tf.placeholder(tf.float32, shape=output_shape)
    gt_lysosome_mem = tf.placeholder(tf.float32, shape=output_shape)

    w_centrosome = tf.placeholder(tf.float32, shape=output_shape)
    w_golgi = tf.placeholder(tf.float32, shape=output_shape)
    w_golgi_mem = tf.placeholder(tf.float32, shape=output_shape)
    w_er = tf.placeholder(tf.float32, shape=output_shape)
    w_er_mem = tf.placeholder(tf.float32, shape=output_shape)
    w_mvb = tf.placeholder(tf.float32, shape=output_shape)
    w_mvb_mem = tf.placeholder(tf.float32, shape=output_shape)
    w_mito = tf.placeholder(tf.float32, shape=output_shape)
    w_mito_mem = tf.placeholder(tf.float32, shape=output_shape)
    w_lysosome = tf.placeholder(tf.float32, shape=output_shape)
    w_lysosome_mem = tf.placeholder(tf.float32, shape=output_shape)

    lb_centrosome = tf.losses.mean_squared_error(gt_centrosome, centrosome, w_centrosome)
    lb_golgi = tf.losses.mean_squared_error(gt_golgi, golgi, w_golgi)
    lb_golgi_mem = tf.losses.mean_squared_error(gt_golgi_mem, golgi_mem, w_golgi_mem)
    lb_er = tf.losses.mean_squared_error(gt_er, er, w_er)
    lb_er_mem = tf.losses.mean_squared_error(gt_er_mem, er_mem, w_er_mem)
    lb_mvb = tf.losses.mean_squared_error(gt_mvb, mvb, w_mvb)
    lb_mvb_mem = tf.losses.mean_squared_error(gt_mvb_mem, mvb_mem, w_mvb_mem)
    lb_mito = tf.losses.mean_squared_error(gt_mito, mito, w_mito)
    lb_mito_mem = tf.losses.mean_squared_error(gt_mito_mem, mito_mem, w_mito_mem)
    lb_lysosome = tf.losses.mean_squared_error(gt_lysosome, lysosome, w_lysosome)
    lb_lysosome_mem = tf.losses.mean_squared_error(gt_lysosome_mem, lysosome_mem, w_lysosome_mem)

    tf.summary.scalar('lb_centrosome', lb_centrosome)
    tf.summary.scalar('lb_golgi', lb_golgi)
    tf.summary.scalar('lb_golgi_mem', lb_golgi_mem)
    tf.summary.scalar('lb_er', lb_er)
    tf.summary.scalar('lb_er_mem', lb_er_mem)
    tf.summary.scalar('lb_mvb', lb_mvb)
    tf.summary.scalar('lb_mvb_mem', lb_mvb_mem)
    tf.summary.scalar('lb_mito', lb_mito)
    tf.summary.scalar('lb_mito_mem', lb_mito_mem)
    tf.summary.scalar('lb_lysosome', lb_lysosome)
    tf.summary.scalar('lb_lysosome_mem', lb_lysosome_mem)

    lub_centrosome = tf.losses.mean_squared_error(gt_centrosome, centrosome)
    lub_golgi = tf.losses.mean_squared_error(gt_golgi, golgi)
    lub_golgi_mem = tf.losses.mean_squared_error(gt_golgi_mem, golgi_mem)
    lub_er = tf.losses.mean_squared_error(gt_er, er)
    lub_er_mem = tf.losses.mean_squared_error(gt_er_mem, er_mem)
    lub_mvb = tf.losses.mean_squared_error(gt_mvb, mvb)
    lub_mvb_mem = tf.losses.mean_squared_error(gt_mvb_mem, mvb_mem)
    lub_mito = tf.losses.mean_squared_error(gt_mito, mito)
    lub_mito_mem = tf.losses.mean_squared_error(gt_mito_mem, mito_mem)
    lub_lysosome = tf.losses.mean_squared_error(gt_lysosome, lysosome)
    lub_lysosome_mem = tf.losses.mean_squared_error(gt_lysosome_mem, lysosome_mem)

    tf.summary.scalar('lub_centrosome', lub_centrosome)
    tf.summary.scalar('lub_golgi', lub_golgi)
    tf.summary.scalar('lub_golgi_mem', lub_golgi_mem)
    tf.summary.scalar('lub_er', lub_er)
    tf.summary.scalar('lub_er_mem', lub_er_mem)
    tf.summary.scalar('lub_mvb', lub_mvb)
    tf.summary.scalar('lub_mvb_mem', lub_mvb_mem)
    tf.summary.scalar('lub_mito', lub_mito)
    tf.summary.scalar('lub_mito_mem', lub_mito_mem)
    tf.summary.scalar('lub_lysosome', lub_lysosome)
    tf.summary.scalar('lub_lysosome_mem', lub_lysosome_mem)


    loss_total = lb_centrosome + lb_golgi + lb_golgi_mem + lb_er + lb_er_mem + lb_mvb + lb_mvb_mem + lb_mito + \
                 lb_mito_mem + lb_lysosome + lb_lysosome_mem
    loss_total_unbalanced = lub_centrosome + lub_golgi + lub_golgi_mem + lub_er + lub_er_mem + lub_mvb + lub_mvb_mem \
                            + lub_mito + lub_mito_mem + lub_lysosome + lub_lysosome_mem
    tf.summary.scalar('loss_total', loss_total)
    tf.summary.scalar('loss_total_unbalanced', loss_total_unbalanced)
    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)

    optimizer = opt.minimize(loss_total)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename='build.meta')

    names = {
        'raw': raw.name,
        'dist': dist_c.name,

        'centrosome': centrosome.name,
        'golgi': golgi.name,
        'golgi_mem': golgi_mem.name,
        'er': er.name,
        'er_mem': er_mem.name,
        'mvb': mvb.name,
        'mvb_mem': mvb_mem.name,
        'mito': mito.name,
        'mito_mem': mito_mem.name,
        'lysosome': lysosome.name,
        'lysosome_mem': lysosome_mem.name,

        'gt_centrosome': gt_centrosome.name,
        'gt_golgi': gt_golgi.name,
        'gt_golgi_mem': gt_golgi_mem.name,
        'gt_er': gt_er.name,
        'gt_er_mem': gt_er_mem.name,
        'gt_mvb': gt_mvb.name,
        'gt_mvb_mem': gt_mvb_mem.name,
        'gt_mito': gt_mito.name,
        'gt_mito_mem': gt_mito_mem.name,
        'gt_lysosome': gt_lysosome.name,
        'gt_lysosome_mem': gt_lysosome_mem.name,

        'w_centrosome': w_centrosome.name,
        'w_golgi': w_golgi.name,
        'w_golgi_mem': w_golgi_mem.name,
        'w_er': w_er.name,
        'w_er_mem': w_er_mem.name,
        'w_mvb': w_mvb.name,
        'w_mvb_mem': w_mvb_mem.name,
        'w_mito': w_mito.name,
        'w_mito_mem': w_mito_mem.name,
        'w_lysosome': w_lysosome.name,
        'w_lysosome_mem': w_lysosome_mem.name,

        'lb_centrosome': lb_centrosome.name,
        'lb_golgi': lb_golgi.name,
        'lb_golgi_mem': lb_golgi_mem.name,
        'lb_er': lb_er.name,
        'lb_er_mem': lb_er_mem.name,
        'lb_mvb': lb_mvb.name,
        'lb_mvb_mem': lb_mvb_mem.name,
        'lb_mito': lb_mito.name,
        'lb_mito_mem': lb_mito_mem.name,
        'lb_lysosome': lb_lysosome.name,
        'lb_lysosome_mem': lb_lysosome_mem.name,

        'lub_centrosome': lub_centrosome.name,
        'lub_golgi': lub_golgi.name,
        'lub_golgi_mem': lub_golgi_mem.name,
        'lub_er': lub_er.name,
        'lub_er_mem': lub_er_mem.name,
        'lub_mvb': lub_mvb.name,
        'lub_mvb_mem': lub_mvb_mem.name,
        'lub_mito': lub_mito.name,
        'lub_mito_mem': lub_mito_mem.name,
        'lub_lysosome': lub_lysosome.name,
        'lub_lysosome_mem': lub_lysosome_mem.name,

        'loss_total': loss_total.name,
        'optimizer': optimizer.name,
        'summary': merged.name}

    with open('net_io_names.json', 'w') as f:
        json.dump(names, f)


def inference_net():
    input_shape = (400, 400, 400)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_bc = tf.reshape(raw, (1, 1,) + input_shape)

    last_fmap, fov, anisotropy = unet.unet(raw_bc, 12, 6, [[2, 2, 2], [2, 2, 2], [3, 3, 3]],
                                           [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=(1, 1, 1), fov=(1, 1, 1))

    dist_bc, fov = ops.conv_pass(
            last_fmap,
            kernel_size=[[1, 1, 1]],
            num_fmaps=11,
            activation=None,
            fov=fov,
            voxel_size=anisotropy
            )
    output_shape_bc = dist_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]  # strip the batch dimension
    output_shape = output_shape_c[1:]

    dist_c = tf.reshape(dist_bc, output_shape_c)
    centrosome, golgi, golgi_mem, er, er_mem, mvb_mem, mvb_mem, mito, mito_mem, lysosome, lysosome_mem = tf.unstack(
        dist_c, 11, axis=0)
    tf.train.export_meta_graph(filename='unet_inference.meta')


if __name__ == '__main__':
    train_net()
    tf.reset_default_graph()
    inference_net()
