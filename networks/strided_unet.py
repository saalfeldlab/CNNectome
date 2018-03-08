from __future__ import print_function
import tensorflow as tf


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


def downsample(fmaps_in, factors, num_fmaps, name='down', fov=(1,1,1), voxel_size=(1, 1, 1), prefix='',
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


def strided_unet(
        fmaps_in,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        activation='relu',
        layer=0,
        fov=(1,1,1),
        voxel_size=(1, 1, 1),
        ):

    '''Create a U-Net::
        f_in --> f_left --------------------------->> f_right--> f_out
                    |                                   ^
                    v                                   |
                 g_in --> g_left ------->> g_right --> g_out
                             |               ^
                             v               |
                                   ...
    where each ``-->`` is a convolution pass (see ``conv_pass``), each `-->>` a
    crop, and down and up arrows are max-pooling and transposed convolutions,
    respectively.
    The U-Net expects tensors to have shape ``(batch=1, channels, depth, height,
    width)``.
    This U-Net performs only "valid" convolutions, i.e., sizes of the feature
    maps decrease after each convolution.
    Args:
        fmaps_in:
            The input tensor.
        num_fmaps:
            The number of feature maps in the first layer. This is also the
            number of output feature maps.
        fmap_inc_factor:
            By how much to multiply the number of feature maps between layers.
            If layer 0 has ``k`` feature maps, layer ``l`` will have
            ``k*fmap_inc_factor**l``.
        downsample_factors:
            List of lists ``[z, y, x]`` to use to down- and up-sample the
            feature maps between layers.
        kernel_size_down:
            List of lists of tuples ``(z, y, x)`` of kernel sizes. The number of
            tuples in a list determines the number of convolutional layers in the
            corresponding level of the unet on the left side.
        kernel_size_up:
            List of lists of tuples ``(z, y, x)`` of kernel sizes. The number of
            tuples in a list determines the number of convolutional layers in the
            corresponding level of the unet on the right side. Within one of the
            lists going from left to right.
        activation:
            Which activation to use after a convolution. Accepts the name of any
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).
        layer:
            Used internally to build the U-Net recursively.
        fov:
            Initial field of view in physical units
        voxel_size:
            Size of a voxel in the input data, in physical units

    '''

    prefix = "    "*layer
    print(prefix + "Creating U-Net layer %i"%layer)
    print(prefix + "f_in: " + str(fmaps_in.shape))

    # convolve
    with tf.name_scope("lev%i"%layer):

        f_left, fov = conv_pass(
            fmaps_in,
            kernel_size=kernel_size_down[layer],
            num_fmaps=num_fmaps,
            activation=activation,
            name='unet_layer_%i_left'%layer,
            fov=fov,
            voxel_size=voxel_size,
            prefix=prefix
            )

        # last layer does not recurse
        bottom_layer = (layer == len(downsample_factors))

        if bottom_layer:
            print(prefix + "bottom layer")
            print(prefix + "f_out: " + str(f_left.shape))
            return f_left, fov, voxel_size

        # downsample

        g_in, fov, voxel_size = downsample(
            f_left,
            downsample_factors[layer],
            num_fmaps=num_fmaps,
            name='unet_down_%i_to_%i'%(layer, layer + 1),
            fov=fov,
            voxel_size=voxel_size,
            prefix=prefix)


        # recursive U-net
        g_out, fov, voxel_size = strided_unet(
            g_in,
            num_fmaps=num_fmaps*fmap_inc_factor,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            activation=activation,
            layer=layer+1,
            fov=fov,
            voxel_size=voxel_size)

        print(prefix + "g_out: " + str(g_out.shape))

        # upsample
        g_out_upsampled, voxel_size = upsample(
            g_out,
            downsample_factors[layer],
            num_fmaps,
            activation=activation,
            name='unet_up_%i_to_%i'%(layer + 1, layer),
            fov=fov,
            voxel_size=voxel_size,
            prefix=prefix)

        print(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

        # copy-crop
        f_left_cropped = crop_zyx(f_left, g_out_upsampled.get_shape().as_list())

        print(prefix + "f_left_cropped: " + str(f_left_cropped.shape))

        # concatenate along channel dimension
        f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)

        print(prefix + "f_right: " + str(f_right.shape))

        # convolve
        f_out,  fov = conv_pass(
            f_right,
            kernel_size=kernel_size_up[layer],
            num_fmaps=num_fmaps,
            name='unet_layer_%i_right'%layer,
            fov=fov,
            voxel_size=voxel_size,
            prefix=prefix
            )

        print(prefix + "f_out: " + str(f_out.shape))

    return f_out, fov, voxel_size


if __name__ == "__main__":
    raw = tf.placeholder(tf.float32, shape=(43, 430, 430))
    raw_batched = tf.reshape(raw, (1, 1,) + (43, 430, 430))

    model, ll_fov, vx = strided_unet(raw_batched,
                             12, 6, [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
                             [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                             [(3, 3, 3), (3, 3, 3)]],
                             [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                             [(3, 3, 3), (3, 3, 3)]],
                             voxel_size=(10, 1, 1), fov=(10, 1, 1))

    output, full_fov = conv_pass(
        model,
        kernel_size=[(1, 1, 1)],
        num_fmaps=1,
        activation=None,
        fov=ll_fov,
        voxel_size=vx
        )

    tf.train.export_meta_graph(filename='unet.meta')

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        tf.summary.FileWriter('.', graph=tf.get_default_graph())

    print(model.shape)