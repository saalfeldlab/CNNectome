import tensorflow as tf
import numpy as np


def conv_pass(
    fmaps_in,
    kernel_size,
    num_fmaps,
    activation="relu",
    name="conv_pass",
    fov=(1, 1),
    voxel_size=(1, 1),
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
        fmaps_pre = tf.layers.conv2d(
            inputs=fmaps,
            filters=num_fmaps,
            kernel_size=ks,
            padding="valid",
            data_format="channels_first",
            activation=activation,
            name=name + "_%i" % i,
        )
    return fmaps, fov


def downsample(
    fmaps_in, factors, name="down", fov=(1, 1), voxel_size=(1, 1), prefix=""
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
    fmaps = tf.layers.max_pooling2d(
        fmaps_in,
        pool_size=factors,
        strides=factors,
        padding="valid",
        data_format="channels_first",
        name=name,
    )
    assert np.sum(np.array(fmaps_in.get_shape()[2:]) % np.array(factors)) == 0
    return fmaps, fov, voxel_size


def downsample_stridedconv(
    fmaps_in,
    factors,
    num_fmaps,
    name="down",
    fov=(1, 1),
    voxel_size=(1, 1),
    prefix="",
    activation="relu",
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
    fmaps = tf.layers.conv2d(
        inputs=fmaps_in,
        filters=num_fmaps,
        kernel_size=factors,
        strides=factors,
        padding="valid",
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
    name="up",
    fov=(1, 1),
    voxel_size=(1, 1),
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
        # (1, 1, num_fmaps_out, num_fmaps_in)
        kernel_variables = tf.reshape(
            kernel_variables, (1, 1) + (num_fmaps_out, num_fmaps_in)
        )
        # (f_y, f_x, num_fmaps_out, num_fmaps_in)
        constant_upsample_filter = repeat(kernel_variables, tuple(factors) + (1, 1))

        fmaps = tf.nn.conv2d_transpose(
            fmaps_in,
            filter=constant_upsample_filter,
            output_shape=out_shape,
            strides=(1, 1) + tuple(factors),
            padding="VALID",
            data_format="NCHW",
            name=name,
        )
        if activation is not None:
            fmaps = activation(fmaps)

    else:

        fmaps = tf.layers.conv2d_transpose(
            fmaps_in,
            filters=num_fmaps,
            kernel_size=factors,
            strides=factors,
            padding="valid",
            data_format="channels_first",
            activation=activation,
            name=name,
        )

    return fmaps, voxel_size


def crop_yx(fmaps_in, shape):
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
        (in_shape[2] - shape[2]) // 2,  # y
        (in_shape[3] - shape[3]) // 2,  # x
    ]
    size = [in_shape[0], in_shape[1], shape[2], shape[3]]

    fmaps = tf.slice(fmaps_in, offset, size)

    return fmaps
