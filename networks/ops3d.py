from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import logging

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
        logging.info(
            prefix + 'fov: {0:} voxsize: {1:} anisotropy: {2:}'.format(fov, voxel_size, (fov[0]) / float(fov[1])))

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
    logging.info(prefix + 'fov: {0:} voxsize: {1:} anisotropy: {2:}'.format(fov, voxel_size, (fov[0]) / float(fov[1])))

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
    logging.info(prefix + 'fov: {0:} voxsize: {1:} anisotropy: {2:}'.format(fov, voxel_size, (fov[0]) / float(fov[1])))

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

def repeat(fmaps_in, multiples):
    expanded = tf.expand_dims(fmaps_in, -1)
    tiled = tf.tile(expanded, multiples = (1,) + multiples)
    repeated = tf.reshape(tiled, tf.shape(fmaps_in) * multiples)
    return repeated

def upsample(fmaps_in, factors, num_fmaps, activation='relu', name='up', fov=(1, 1, 1), voxel_size=(1, 1, 1),
             prefix='', constant_upsample=False):

    voxel_size = tuple(vs / fac for vs, fac in zip(voxel_size, factors))

    logging.info(prefix + 'fov: {0:} voxsize: {1:} anisotropy: {2:}'.format(fov, voxel_size, (fov[0]) / float(fov[1])))
    if activation is not None:
        activation = getattr(tf.nn, activation)

    if constant_upsample:
        in_shape = tuple(fmaps_in.get_shape().as_list())
        num_fmaps_in = in_shape[1]
        num_fmaps_out = num_fmaps
        out_shape = (in_shape[0], num_fmaps_out) + tuple(s * f for s, f in zip(in_shape[2:], factors))

        # (num_fmaps_out * num_fmaps_in)
        kernel_variables = tf.get_variable(name + '_kernel_variables', (num_fmaps_out * num_fmaps_in,),
                                           dtype=tf.float32)
        # (1, 1, 1, num_fmaps_out, num_fmaps_in)
        kernel_variables = tf.reshape(kernel_variables, (1, 1, 1) + (num_fmaps_out, num_fmaps_in))
        # (f_z, f_y, f_x, num_fmaps_out, num_fmaps_in)
        constant_upsample_filter = repeat(kernel_variables, tuple(factors) + (1, 1))

        fmaps = tf.nn.conv3d_transpose(fmaps_in,
                                       filter=constant_upsample_filter,
                                       output_shape=out_shape,
                                       strides=(1, 1) + tuple(factors),
                                       padding='VALID',
                                       data_format='NCDHW',
                                       name=name)
        if activation is not None:
            fmaps = activation(fmaps)

    else:
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


def crop_to_factor(fmaps_in, factor, kernel_sizes):
    '''Crop feature maps to ensure translation equivariance with stride of
    upsampling factor. This should be done right after upsampling, before
    application of the convolutions with the given kernel sizes.
    The crop could be done after the convolutions, but it is more efficient to
    do that before (feature maps will be smaller).
    '''

    shape = fmaps_in.get_shape().as_list()
    spatial_dims = 3 if len(shape) == 5 else 4
    spatial_shape = shape[-spatial_dims:]

    # the crop that will already be done due to the convolutions
    convolution_crop = list(
        sum(
            (ks if isinstance(ks, int) else ks[d]) - 1
            for ks in kernel_sizes
        )
        for d in range(spatial_dims)
    )
    logging.debug("crop_to_factor: factor =", factor)
    logging.debug("crop_to_factor: kernel_sizes =", kernel_sizes)
    logging.debug("crop_to_factor: convolution_crop =", convolution_crop)

    # we need (spatial_shape - convolution_crop) to be a multiple of factor,
    # i.e.:
    #
    # (s - c) = n*k
    #
    # we want to find the largest n for which s' = n*k + c <= s
    #
    # n = floor((s - c)/k)
    #
    # this gives us the target shape s'
    #
    # s' = n*k + c

    ns = (
        int(math.floor(float(s - c)/f))
        for s, c, f in zip(spatial_shape, convolution_crop, factor)
    )
    target_spatial_shape = tuple(
        n*f + c
        for n, c, f in zip(ns, convolution_crop, factor)
    )
    if target_spatial_shape != spatial_shape:

        assert all((
                (t > c) for t, c in zip(
                    target_spatial_shape,
                    convolution_crop))
            ), \
            "Feature map with shape %s is too small to ensure translation " \
            "equivariance with factor %s and following convolutions %s" % (
                shape,
                factor,
                kernel_sizes)

        target_shape = list(shape)
        target_shape[-spatial_dims:] = target_spatial_shape

        logging.debug("crop_to_factor: shape = {0:}".format(shape))
        logging.debug("crop_to_factor: spatial_shape = {0:}".format(spatial_shape))
        logging.debug("crop_to_factor: target_spatial_shape = {0:}".format(target_spatial_shape))
        logging.debug("crop_to_factor: target_shape ={0:}".format(target_shape))
        fmaps = crop_zyx(
            fmaps_in,
            target_shape)
    else:
        fmaps = fmaps_in

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

    logging.debug("Cropping from %s to %s"%(shape, size))
    logging.debug("Diff: %s"%(diff,))
    logging.debug("Slices: %s"%(slices,))

    cropped = tensor[slices]

    logging.debug("Result size: %s"%cropped.get_shape().as_list())

    return cropped