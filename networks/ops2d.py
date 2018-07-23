from __future__ import print_function
import tensorflow as tf


def conv_pass(
        fmaps_in,
        kernel_size,
        num_fmaps,
        activation='relu',
        name='conv_pass',
        fov=(1, 1),
        voxel_size=(1, 1),
        prefix=''):
    '''Create a convolution pass::
        f_in --> f_1 --> ... --> f_n
    where each ``-->`` is a convolution followed by a (non-linear) activation
    function and ``n`` ``num_repetitions``. Each convolution will decrease the
    size of the feature maps by ``kernel_size-1``.
    Args:
        f_in:
            The input tensor of shape ``(batch_size, channels, depth, height, width)``.
        kernel_size:
            List of sizes of kernels. Length determines number of convolutional layers.
            Kernel size forwarded to tf.layers.conv3d.
        num_fmaps:
            The number of feature maps to produce with each convolution.
        activation:
            Which activation to use after a convolution. Accepts the name of any
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).
        name:
            Base name for the conv layer.
        fov:
            Field of view of fmaps_in, in physical units.
        voxel_size:
            Size of a voxel in the input data, in physical units.


    '''

    fmaps = fmaps_in
    if activation is not None:
        activation = getattr(tf.nn, activation)

    for i, ks in enumerate(kernel_size):
        fov = tuple(f + (k-1) * vs for f, k, vs in zip(fov, ks, voxel_size))
        print(prefix, 'fov:', fov, 'voxsize:', voxel_size, 'anisotropy:', (fov[0]) / float(fov[1]))
        fmaps_pre = tf.layers.conv2d(
            inputs=fmaps,
            filters=num_fmaps,
            kernel_size=ks,
            padding='valid',
            data_format='channels_first',
            activation=None,
            name=name + '_%i'%i)
        fmaps = tf.nn.relu(fmaps_pre)
    return fmaps, fov


def downsample(fmaps_in, factors, name='down', fov=(1,1), voxel_size=(1, 1), prefix=''):
    #fov = [f+(fac-1)*ai for f, fac,ai in zip(fov, factors,anisotropy)]
    voxel_size = tuple(vs * fac for vs, fac in zip(voxel_size, factors))
    print(prefix, 'fov:', fov, 'voxsize:', voxel_size, 'anisotropy:', (fov[0]) / float(fov[1]))
    fmaps = tf.layers.max_pooling2d(
        fmaps_in,
        pool_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        name=name)

    return fmaps, fov, voxel_size


def upsample(fmaps_in, factors, num_fmaps, activation='relu', name='up', fov=(1, 1), voxel_size=(1, 1),
             prefix=''):

    voxel_size = tuple(vs / fac for vs, fac in zip(voxel_size, factors))

    print(prefix, 'fov:', fov, 'voxsize:', voxel_size, 'anisotropy:', (fov[0]) / float(fov[1]))
    if activation is not None:
        activation = getattr(tf.nn, activation)

    fmaps = tf.layers.conv2d_transpose(
        fmaps_in,
        filters=num_fmaps,
        kernel_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        activation=activation,
        name=name)

    return fmaps, voxel_size


def crop_yx(fmaps_in, shape):
    '''Crop only the spacial dimensions to match shape.
    Args:
        fmaps_in:
            The input tensor.
        shape:
            A list (not a tensor) with the requested shape [_, _, z, y, x].
    '''

    in_shape = fmaps_in.get_shape().as_list()

    offset = [
        0, # batch
        0, # channel
        (in_shape[2] - shape[2])//2, # y
        (in_shape[3] - shape[3])//2, # x
    ]
    size = [
        in_shape[0],
        in_shape[1],
        shape[2],
        shape[3]
    ]

    fmaps = tf.slice(fmaps_in, offset, size)

    return fmaps
