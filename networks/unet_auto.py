import tensorflow as tf
import numpy as np
import ops3d
import warnings

def unet_auto(
        fmaps_in,
        pmaps_in,
        num_fmaps,
        fmap_inc_factors,
        downsample_factors,
        fkernel_size_down,
        pkernel_size_down,
        kernel_size_up,
        activation='relu',
        layer=0,
        fov=(1, 1, 1),
        voxel_size=(1, 1, 1)
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
        pmaps_in:
            TODO
        num_fmaps:
            The number of feature maps in the first layer. This is also the
            number of output feature maps.
        fmap_inc_factors:
            By how much to multiply the number of feature maps between layers.
            If layer 0 has ``k`` feature maps, layer ``l`` will have
            ``k*fmap_inc_factor**l``.
        downsample_factors:
            List of lists ``[z, y, x]`` to use to down- and up-sample the
            feature maps between layers.
        kernel_size_down:
            List of lists of tuples ``(z, y, x)`` of kernel sizes. The number of
            tuples in a list determines the number of convolutional layers in the
            corresponding level of the build on the left side.
        kernel_size_up:
            List of lists of tuples ``(z, y, x)`` of kernel sizes. The number of
            tuples in a list determines the number of convolutional layers in the
            corresponding level of the build on the right side. Within one of the
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
    if isinstance(fmap_inc_factors, int):
        fmap_inc_factors = [fmap_inc_factors] * len(downsample_factors)
    assert len(fmap_inc_factors) == len(downsample_factors) == len(fkernel_size_down) - 1 == len(pkernel_size_down) \
                                                                                             -1==len(kernel_size_up) - 1
    if layer == 0:
        warnings.warn('FOV calculation does not respect autocontext yet')
    # convolve
    f_left, fov = ops3d.conv_pass(
        fmaps_in,
        kernel_size=fkernel_size_down[layer],
        num_fmaps=num_fmaps,
        activation=activation,
        name='unet_layer_%i_left'%layer,
        fov=fov,
        voxel_size=voxel_size,
        prefix=prefix
    )

    p_left, _ = ops3d.conv_pass(
        pmaps_in,
        kernel_size=pkernel_size_down[layer],
        num_fmaps=num_fmaps,
        activation=activation,
        name='unet_layer_pred_%i_left'%layer,
        fov=fov,
        voxel_size=voxel_size,
        prefix=prefix)

    fp_left = ops3d.crossmod_conv_pass(
        f_left, p_left,
        num_fmaps=num_fmaps,
        activation=activation,
        name='unet_layer_%i_crossmod'%layer)


    # last layer does not recurse
    bottom_layer = (layer == len(downsample_factors))
    if bottom_layer:
        print(prefix + "bottom layer")
        print(prefix + "f_out: " + str(f_left.shape))
        return fp_left, fov, voxel_size

    # downsample
    g_in,fov,voxel_size = ops3d.downsample(
        fp_left,
        downsample_factors[layer],
        'unet_down_%i_to_%i'%(layer, layer + 1),
        fov=fov,
        voxel_size=voxel_size,
        prefix=prefix)

    q_in, _, _ = ops3d.downsample(
        p_left,
        downsample_factors[layer],
        'unet_down_pred_%i_to_%i'%(layer, layer + 1))

    # recursive U-net
    g_out, fov, voxel_size = unet_auto(
        g_in,
        q_in,
        num_fmaps=num_fmaps*fmap_inc_factors[layer],
        fmap_inc_factors=fmap_inc_factors,
        downsample_factors=downsample_factors,
        fkernel_size_down=fkernel_size_down,
        pkernel_size_down=pkernel_size_down,
        kernel_size_up=kernel_size_up,
        activation=activation,
        layer=layer+1,
        fov=fov,
        voxel_size=voxel_size)

    print(prefix + "g_out: " + str(g_out.shape))

    # upsample
    g_out_upsampled, voxel_size = ops3d.upsample(
        g_out,
        downsample_factors[layer],
        num_fmaps,
        activation=activation,
        name='unet_up_%i_to_%i'%(layer + 1, layer),
        fov=fov,
        voxel_size=voxel_size,
        prefix=prefix
    )

    print(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

    # copy-crop
    fp_left_cropped = ops3d.crop_zyx(fp_left, g_out_upsampled.get_shape().as_list())

    print(prefix + "fp_left_cropped: " + str(fp_left_cropped.shape))

    # concatenate along channel dimension
    f_right = tf.concat([fp_left_cropped, g_out_upsampled], 1)

    print(prefix + "f_right: " + str(f_right.shape))

    # convolve
    f_out, fov = ops3d.conv_pass(
        f_right,
        kernel_size=kernel_size_up[layer],
        num_fmaps=num_fmaps,
        name='unet_layer_%i_right'%layer,
        fov=fov,
        voxel_size=voxel_size,
        prefix=prefix)

    print(prefix + "f_out: " + str(f_out.shape))

    return f_out, fov, voxel_size
