from networks import unet, ops3d
import tensorflow as tf
import json



def train_net():
    input_shape = (196, 196, 196)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_bc = tf.reshape(raw, (1, 1,)+input_shape)

    last_fmap, fov, anisotropy = unet.unet(raw_bc, 12, 6, [[2, 2, 2], [2, 2, 2], [3, 3, 3]],
                                           [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=(1, 1, 1), fov=(1, 1, 1))
    pred_raw_bc, fov = ops3d.conv_pass(
        last_fmap,
        kernel_size=[[1,1,1]],
        num_fmaps=1,
        activation=None,
        fov=fov,
        voxel_size=anisotropy
    )
    output_shape_bc = pred_raw_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]
    output_shape = output_shape_c[1:]

    pred_raw = tf.reshape(pred_raw_bc, output_shape)

    gt_raw_bc = ops3d.crop_zyx(raw_bc, output_shape_bc)
    gt_raw = tf.reshape(gt_raw_bc, output_shape)

    loss = tf.losses.mean_squared_error(gt_raw, pred_raw)
    tf.summary.scalar('loss', loss)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8
    )
    optimizer = opt.minimize(loss)

    merged = tf.summary.merge_all()
    tf.train.export_meta_graph(filename='unet.meta')

    names={
        'raw': raw.name,
        'pred_raw': pred_raw.name,
        'optimizer': optimizer.name,
        'summary': merged.name,
        'loss': loss.name
    }
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
    pred_raw_bc, fov = ops3d.conv_pass(
        last_fmap,
        kernel_size=[[1, 1, 1]],
        num_fmaps=1,
        activation=None,
        fov=fov,
        voxel_size=anisotropy
    )
    output_shape_bc = pred_raw_bc.get_shape().as_list()
    output_shape_c = output_shape_bc[1:]
    output_shape = output_shape_c[1:]

    pred_raw = tf.reshape(pred_raw_bc, output_shape)

    tf.train.export_meta_graph(filename='unet_inference.meta')

if __name__ == '__main__':
    train_net()
    tf.reset_default_graph()
    inference_net()