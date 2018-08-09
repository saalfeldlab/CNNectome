from __future__ import print_function
import tensorflow as tf
import numpy as np

def conv_pass(
        fmaps_in,
        kernel_size,
        num_fmaps,
        activation='relu',
        name='conv_pass',
        fov=(1, 1, 1),
        voxel_size=(1, 1, 1),
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
        fmaps = tf.layers.conv3d(
            inputs=fmaps,
            filters=num_fmaps,
            kernel_size=ks,
            padding='valid',
            data_format='channels_first',
            activation=activation,
            name=name + '_%i'%i)

    return fmaps, fov


def downsample(fmaps_in, factors, name='down', fov=(1,1,1), voxel_size=(1, 1, 1), prefix=''):
    #fov = [f+(fac-1)*ai for f, fac,ai in zip(fov, factors,anisotropy)]
    voxel_size = tuple(vs * fac for vs, fac in zip(voxel_size, factors))
    print(prefix, 'fov:', fov, 'voxsize:', voxel_size, 'anisotropy:', (fov[0]) / float(fov[1]))
    fmaps = tf.layers.max_pooling3d(
        fmaps_in,
        pool_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        name=name)
    assert np.sum(np.array(fmaps_in.get_shape()[2:])%np.array(factors))==0
    return fmaps, fov, voxel_size


def downsample_stridedconv(fmaps_in, factors, num_fmaps, name='down', fov=(1,1,1), voxel_size=(1, 1, 1), prefix='',
               activation='relu'):
    #fov = [f+(fac-1)*ai for f, fac,ai in zip(fov, factors,anisotropy)]
    if activation is not None:
        activation = getattr(tf.nn, activation)
    voxel_size = tuple(vs * fac for vs, fac in zip(voxel_size, factors))
    print(prefix, 'fov:', fov, 'voxsize:', voxel_size, 'anisotropy:', (fov[0]) / float(fov[1]))
    fmaps = tf.layers.conv3d(
        inputs=fmaps_in,
        filters=num_fmaps,
        kernel_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        activation=activation,
        name=name)

    return fmaps, fov, voxel_size


def upsample(fmaps_in, factors, num_fmaps, activation='relu', name='up', fov=(1, 1, 1), voxel_size=(1, 1, 1),
             prefix=''):

    voxel_size = tuple(vs / fac for vs, fac in zip(voxel_size, factors))

    print(prefix, 'fov:', fov, 'voxsize:', voxel_size, 'anisotropy:', (fov[0]) / float(fov[1]))
    if activation is not None:
        activation = getattr(tf.nn, activation)

    fmaps = tf.layers.conv3d_transpose(
        fmaps_in,
        filters=num_fmaps,
        kernel_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        activation=activation,
        name=name)

    return fmaps, voxel_size


def crop_zyx(fmaps_in, shape):
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
        (in_shape[2] - shape[2])//2, # z
        (in_shape[3] - shape[3])//2, # y
        (in_shape[4] - shape[4])//2, # x
    ]
    size = [
        in_shape[0],
        in_shape[1],
        shape[2],
        shape[3],
        shape[4],
    ]

    fmaps = tf.slice(fmaps_in, offset, size)

    return fmaps


def crossmod_conv_pass(
        fmaps_in,
        pmaps_in,
        num_fmaps,
        activation='relu',
        name='crossmod_conv_pass'):
    in_ch, z = fmaps_in.get_shape().as_list()[1:3]
    in_pred_ch, z_pred = pmaps_in.get_shape().as_list()[1:3]
    assert in_ch == in_pred_ch
    assert z == z_pred

    fp_maps = tf.concat([fmaps_in, pmaps_in], 2)
    num_mods = fp_maps.get_shape().as_list()[2]/z
    f = tf.get_variable('crossmod_filter_of_' + name, (num_mods, 1, 1, in_ch, num_fmaps), trainable=True)
    return tf.nn.convolution(input=fp_maps, filter=f, padding='VALID', strides=[1, 1, 1],
                      dilation_rate=[z, 1, 1], data_format='NCDHW', name=name)


def center_crop(tensor, size):

    shape = tensor.get_shape().as_list()
    diff = tuple(sh - si for sh, si in zip(shape, size))

    for d in diff:
        assert d >= 0
        assert d%2 == 0

    slices = tuple(slice(d/2, -d/2) if d > 0 else slice(None) for d in diff)

    print("Cropping from %s to %s"%(shape, size))
    print("Diff: %s"%(diff,))
    print("Slices: %s"%(slices,))

    cropped = tensor[slices]

    print("Result size: %s"%cropped.get_shape().as_list())

    return cropped