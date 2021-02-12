import tensorflow.compat.v1 as tf
import numpy as np
import math
import logging


def conv_pass(
    fmaps_in,
    kernel_size,
    num_fmaps,
    activation="relu",
    padding="valid",
    name="conv_pass",
    fov=(1, 1, 1),
    voxel_size=(1, 1, 1),
    prefix="",
):
    """Create a convolution pass::
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


    """

    fmaps = fmaps_in
    if activation is not None:
        activation = getattr(tf.nn, activation)

    for i, ks in enumerate(kernel_size):
        fov = tuple(f + (k - 1) * vs for f, k, vs in zip(fov, ks, voxel_size))
        print(
            prefix,
            "fov:",
            fov,
            "voxsize:",
            voxel_size,
            "anisotropy:",
            (fov[0]) / float(fov[1]),
        )
        fmaps = tf.layers.conv3d(
            inputs=fmaps,
            filters=num_fmaps,
            kernel_size=ks,
            padding=padding,
            data_format="channels_first",
            activation=activation,
            name=name + "_%i" % i,
        )

    return fmaps, fov


def downsample(
    fmaps_in,
    factors,
    padding="valid",
    name="down",
    fov=(1, 1, 1),
    voxel_size=(1, 1, 1),
    prefix="",
):
    # fov = [f+(fac-1)*ai for f, fac,ai in zip(fov, factors,anisotropy)]
    voxel_size = tuple(vs * fac for vs, fac in zip(voxel_size, factors))
    print(
        prefix,
        "fov:",
        fov,
        "voxsize:",
        voxel_size,
        "anisotropy:",
        (fov[0]) / float(fov[1]),
    )
    fmaps = tf.layers.max_pooling3d(
        fmaps_in,
        pool_size=factors,
        strides=factors,
        padding=padding,
        data_format="channels_first",
        name=name,
    )
    if padding == "valid":
        assert int(np.sum(np.array(fmaps_in.get_shape()[2:]) % np.array(factors))) == 0
    return fmaps, fov, voxel_size


def downsample_stridedconv(
    fmaps_in,
    factors,
    num_fmaps,
    activation="relu",
    padding="valid",
    name="down",
    fov=(1, 1, 1),
    voxel_size=(1, 1, 1),
    prefix="",
):
    # fov = [f+(fac-1)*ai for f, fac,ai in zip(fov, factors,anisotropy)]
    if activation is not None:
        activation = getattr(tf.nn, activation)
    voxel_size = tuple(vs * fac for vs, fac in zip(voxel_size, factors))
    print(
        prefix,
        "fov:",
        fov,
        "voxsize:",
        voxel_size,
        "anisotropy:",
        (fov[0]) / float(fov[1]),
    )
    fmaps = tf.layers.conv3d(
        inputs=fmaps_in,
        filters=num_fmaps,
        kernel_size=factors,
        strides=factors,
        padding=padding,
        data_format="channels_first",
        activation=activation,
        name=name,
    )

    return fmaps, fov, voxel_size


def repeat(fmaps_in, multiples):
    expanded = tf.expand_dims(fmaps_in, -1)
    tiled = tf.tile(expanded, multiples=(1,) + multiples)
    repeated = tf.reshape(tiled, tf.shape(fmaps_in) * multiples)
    return repeated


def upsample(
    fmaps_in,
    factors,
    num_fmaps,
    activation="relu",
    padding="valid",
    name="up",
    fov=(1, 1, 1),
    voxel_size=(1, 1, 1),
    prefix="",
    constant_upsample=False,
):

    voxel_size = tuple(vs / fac for vs, fac in zip(voxel_size, factors))

    print(
        prefix,
        "fov:",
        fov,
        "voxsize:",
        voxel_size,
        "anisotropy:",
        (fov[0]) / float(fov[1]),
    )
    if activation is not None:
        activation = getattr(tf.nn, activation)

    if constant_upsample:
        in_shape = tuple(fmaps_in.get_shape().as_list())
        num_fmaps_in = in_shape[1]
        num_fmaps_out = num_fmaps
        out_shape = (in_shape[0], num_fmaps_out) + tuple(
            s * f for s, f in zip(in_shape[2:], factors)
        )

        # (num_fmaps_out * num_fmaps_in)
        kernel_variables = tf.get_variable(
            name + "_kernel_variables",
            (num_fmaps_out * num_fmaps_in,),
            dtype=tf.float32,
        )
        # (1, 1, 1, num_fmaps_out, num_fmaps_in)
        kernel_variables = tf.reshape(
            kernel_variables, (1, 1, 1) + (num_fmaps_out, num_fmaps_in)
        )
        # (f_z, f_y, f_x, num_fmaps_out, num_fmaps_in)
        constant_upsample_filter = repeat(kernel_variables, tuple(factors) + (1, 1))

        fmaps = tf.nn.conv3d_transpose(
            fmaps_in,
            filter=constant_upsample_filter,
            output_shape=out_shape,
            strides=(1, 1) + tuple(factors),
            padding=padding.upper(),
            data_format="NCDHW",
            name=name,
        )
        if activation is not None:
            fmaps = activation(fmaps)

    else:
        fmaps = tf.layers.conv3d_transpose(
            fmaps_in,
            filters=num_fmaps,
            kernel_size=factors,
            strides=factors,
            padding=padding,
            data_format="channels_first",
            activation=activation,
            name=name,
        )

    return fmaps, voxel_size


def gaussian_blur(fmaps_in, sigma):
    def gauss_kernel(sigma):
        if sigma < 0:
            kernel_1d = np.array([0., 1., 0.], dtype=np.float32)
        else:
            kernel_size = max(3, 2 * int(3 * sigma + 0.5) + 1)
            kernel_1d = np.zeros(kernel_size, dtype=np.float32)
            x = int(kernel_size / 2)
            while x >= 0:
                val = np.exp(-x**2/(2*sigma*sigma))
                kernel_1d[int(kernel_size/2) - x] = val
                kernel_1d[int(kernel_size/2) + x] = val
                x -= 1
        kernel = kernel_1d[:,None,None]* kernel_1d[None,:,None]* kernel_1d[None,None,:]
        kernel /= np.sum(kernel)
        return kernel
    gaussian_kernel = tf.convert_to_tensor(gauss_kernel(sigma))
    gaussian_kernel = gaussian_kernel[..., tf.newaxis, tf.newaxis]
    fmaps = tf.nn.conv3d(fmaps_in, gaussian_kernel, padding="SAME", data_format="NCDHW", strides=[1,1,1,1,1], name="gaussian_blur")
    return fmaps


def gaussian_blur_var(fmaps_in, sigma):
    def gauss_kernel(sigma):
        def identity():
            kernel_1d = np.array([0., 1., 0.], dtype=np.float32)
            return kernel_1d
        def construct(sigma):
            kernel_size = tf.math.maximum(3, 2 * tf.cast(3 * sigma + 0.5, tf.int32) + 1)
            x = tf.cast(tf.cast(kernel_size / 2, tf.int32), tf.float32)
            kernel_1d = tf.range(-x, x, dtype=np.float32)
            # def while_cond(x, kernel_1d):
            #     return tf.math.greater_equal(x, 0)
            # def while_body(x, kernel_1d):
            #     val = tf.math.exp(-tf.cast(x, tf.float32)**2/(2*sigma * sigma))
            #     kernel_1d[tf.cast(kernel_size/2, tf.int32) - x] = val
            #     kernel_1d[tf.cast(kernel_size/2, tf.int32) + x] = val
            #     x -= 1
            #     return y

            # x, kernel_1d = tf.while_loop(while_cond, while_body, (x, kernel_1d), return_same_structure=True)
            kernel_1d = tf.math.exp(-kernel_1d**2/(2*sigma*sigma))
            return kernel_1d
        kernel_1d = tf.cond(tf.math.greater(0., sigma), identity, lambda: construct(sigma))
        kernel = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
        kernel /= np.sum(kernel)
        return kernel
    gaussian_kernel = tf.convert_to_tensor(gauss_kernel(sigma))
    gaussian_kernel = gaussian_kernel[..., tf.newaxis, tf.newaxis]
    fmaps = tf.nn.conv3d(fmaps_in, gaussian_kernel, padding="SAME", data_format="NCDHW", strides=[1,1,1,1,1], name="gaussian_blur")
    return fmaps


def crop_zyx(fmaps_in, shape):
    """Crop only the spacial dimensions to match shape.
    Args:
        fmaps_in:
            The input tensor.
        shape:
            A list (not a tensor) with the requested shape [_, _, z, y, x].
    """

    in_shape = fmaps_in.get_shape().as_list()

    offset = [
        0,  # batch
        0,  # channel
        (in_shape[2] - shape[2]) // 2,  # z
        (in_shape[3] - shape[3]) // 2,  # y
        (in_shape[4] - shape[4]) // 2,  # x
    ]
    size = [in_shape[0], in_shape[1], shape[2], shape[3], shape[4]]

    fmaps = tf.slice(fmaps_in, offset, size)

    return fmaps


def crop_to_factor(fmaps_in, factor, kernel_sizes):
    """Crop feature maps to ensure translation equivariance with stride of
    upsampling factor. This should be done right after upsampling, before
    application of the convolutions with the given kernel sizes.
    The crop could be done after the convolutions, but it is more efficient to
    do that before (feature maps will be smaller).
    """

    shape = fmaps_in.get_shape().as_list()
    spatial_dims = 3 if len(shape) == 5 else 4
    spatial_shape = shape[-spatial_dims:]

    # the crop that will already be done due to the convolutions
    convolution_crop = list(
        sum((ks if isinstance(ks, int) else ks[d]) - 1 for ks in kernel_sizes)
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
        int(math.floor(float(s - c) / f))
        for s, c, f in zip(spatial_shape, convolution_crop, factor)
    )
    target_spatial_shape = tuple(
        n * f + c for n, c, f in zip(ns, convolution_crop, factor)
    )
    if target_spatial_shape != spatial_shape:

        assert all(((t > c) for t, c in zip(target_spatial_shape, convolution_crop))), (
            "Feature map with shape %s is too small to ensure translation "
            "equivariance with factor %s and following convolutions %s"
            % (shape, factor, kernel_sizes)
        )

        target_shape = list(shape)
        target_shape[-spatial_dims:] = target_spatial_shape

        logging.debug("crop_to_factor: shape = {0:}".format(shape))
        logging.debug("crop_to_factor: spatial_shape = {0:}".format(spatial_shape))
        logging.debug(
            "crop_to_factor: target_spatial_shape = {0:}".format(target_spatial_shape)
        )
        logging.debug("crop_to_factor: target_shape ={0:}".format(target_shape))
        fmaps = crop_zyx(fmaps_in, target_shape)
    else:
        fmaps = fmaps_in

    return fmaps


def crossmod_conv_pass(
    fmaps_in,
    pmaps_in,
    num_fmaps,
    activation="relu",
    padding="valid",
    name="crossmod_conv_pass",
):
    in_ch, z = fmaps_in.get_shape().as_list()[1:3]
    in_pred_ch, z_pred = pmaps_in.get_shape().as_list()[1:3]
    assert in_ch == in_pred_ch
    assert z == z_pred

    fp_maps = tf.concat([fmaps_in, pmaps_in], 2)
    num_mods = fp_maps.get_shape().as_list()[2] / z
    f = tf.get_variable(
        "crossmod_filter_of_" + name, (num_mods, 1, 1, in_ch, num_fmaps), trainable=True
    )
    return tf.nn.convolution(
        input=fp_maps,
        filter=f,
        padding=padding.upper(),
        strides=[1, 1, 1],
        dilation_rate=[z, 1, 1],
        data_format="NCDHW",
        name=name,
    )


def center_crop(tensor, size):

    shape = tensor.get_shape().as_list()
    diff = tuple(sh - si for sh, si in zip(shape, size))

    for d in diff:
        assert d >= 0
        assert d % 2 == 0

    slices = tuple(slice(d / 2, -d / 2) if d > 0 else slice(None) for d in diff)

    print("Cropping from %s to %s" % (shape, size))
    print("Diff: %s" % (diff,))
    print("Slices: %s" % (slices,))

    cropped = tensor[slices]

    print("Result size: %s" % cropped.get_shape().as_list())

    return cropped
