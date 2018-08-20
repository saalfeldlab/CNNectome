from __future__ import print_function
import tensorflow as tf
import numpy as np
import ops3d
import warnings


class UNet(object):

    def __init__(self, num_fmaps, fmap_inc_factors, downsample_factors, kernel_size_down, kernel_size_up,
                 activation='relu', input_fov=(1, 1, 1), input_voxel_size=(1, 1, 1)):
        if isinstance(fmap_inc_factors, int):
            fmap_inc_factors = [fmap_inc_factors] * len(downsample_factors)
        assert len(fmap_inc_factors) == len(downsample_factors) == len(kernel_size_down) - 1
        assert len(downsample_factors) == len(kernel_size_up) - 1 or len(downsample_factors) == len(kernel_size_up)
        if len(downsample_factors) == len(kernel_size_up) - 1:
            warnings.warn("kernel sizes for upscaling are not used for the bottom layer")
            kernel_size_up = kernel_size_up[:-1]
        self.num_fmaps = num_fmaps
        self.fmap_inc_factors = fmap_inc_factors
        self.downsample_factors = downsample_factors
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up = kernel_size_up
        self.activation = activation
        self.input_fov = input_fov
        self.input_voxel_size = input_voxel_size

    def compute_valid_input_shape(self):
        min_bottom = (0., 0., 0.)
        for lv in range(len(self.downsample_factors)):
            kernels = self.kernel_size_up[lv]
            conv_pad = np.sum([np.array(k) - np.array((1.,1.,1.)) for k in kernels] , axis=0)
            min_bottom += conv_pad/np.prod(self.downsample_factors[lv:], axis=0)
        min_bottom = np.ceil(min_bottom)
        min_input_size = min_bottom

        for lv in range(len(self.kernel_size_up))[::-1]:
            if lv != len(self.kernel_size_up):
                min_input_size *= self.downsample_factors[lv]
            kernels = self.kernel_size_down[lv]
            conv_pad = np.sum([np.array(k) - np.array((1.,1.,1.)) for k in kernels], axis=0)
            min_input_size += conv_pad
            if lv != len(self.kernel_size_up):
                min_bottom_size = min_input_size

        step = np.prod(self.downsample_factors, axis=0)

        return min_input_size.astype(np.int), step


    def unet(self,
             fmaps_in,
             num_fmaps=None,
             fmap_inc_factors=None,
             downsample_factors=None,
             kernel_size_down=None,
             kernel_size_up=None,
             activation=None,
             layer=0,
             fov=None,
             voxel_size=None,
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
        if num_fmaps is None:
            num_fmaps = self.num_fmaps
        if fmap_inc_factors is None:
            fmap_inc_factors = self.fmap_inc_factors
        if downsample_factors is None:
            downsample_factors = self.downsample_factors
        if kernel_size_down is None:
            kernel_size_down = self.kernel_size_down
        if kernel_size_up is None:
            kernel_size_up = self.kernel_size_up
        if activation is None:
            activation = self.activation
        if fov is None:
            fov = self.input_fov
        if voxel_size is None:
            voxel_size = self.input_voxel_size

        prefix = "    " * layer
        print(prefix + "Creating U-Net layer %i" % layer)
        print(prefix + "f_in: " + str(fmaps_in.shape))

        # convolve
        with tf.name_scope("lev%i" % layer):

            f_left, fov = ops3d.conv_pass(
                fmaps_in,
                kernel_size=kernel_size_down[layer],
                num_fmaps=num_fmaps,
                activation=activation,
                name='unet_layer_%i_left' % layer,
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

            g_in, fov, voxel_size = ops3d.downsample(
                f_left,
                downsample_factors[layer],
                'unet_down_%i_to_%i' % (layer, layer + 1),
                fov=fov,
                voxel_size=voxel_size,
                prefix=prefix)

            # recursive U-net
            g_out, fov, voxel_size = self.unet(
                g_in,
                num_fmaps=num_fmaps * fmap_inc_factors[layer],
                fmap_inc_factors=fmap_inc_factors,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                activation=activation,
                layer=layer + 1,
                fov=fov,
                voxel_size=voxel_size)

            print(prefix + "g_out: " + str(g_out.shape))

            # upsample
            g_out_upsampled, voxel_size = ops3d.upsample(
                g_out,
                downsample_factors[layer],
                num_fmaps,
                activation=activation,
                name='unet_up_%i_to_%i' % (layer + 1, layer),
                fov=fov,
                voxel_size=voxel_size,
                prefix=prefix)

            print(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

            # copy-crop
            f_left_cropped = ops3d.crop_zyx(f_left, g_out_upsampled.get_shape().as_list())

            print(prefix + "f_left_cropped: " + str(f_left_cropped.shape))

            # concatenate along channel dimension
            f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)

            print(prefix + "f_right: " + str(f_right.shape))

            # convolve
            f_out, fov = ops3d.conv_pass(
                f_right,
                kernel_size=kernel_size_up[layer],
                num_fmaps=num_fmaps,
                name='unet_layer_%i_right' % layer,
                fov=fov,
                voxel_size=voxel_size,
                prefix=prefix
            )

            print(prefix + "f_out: " + str(f_out.shape))

        return f_out, fov, voxel_size



if __name__ == "__main__":
    model = UNet(
                             12, 6, [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
                             [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                             [(3, 3, 3), (3, 3, 3)]],
                             [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)], [(3, 3, 3), (3, 3, 3)],
                             [(3, 3, 3), (3, 3, 3)]],
                             input_voxel_size=(10, 1, 1), input_fov=(10, 1, 1))
    min_shape, stepsize = model.compute_valid_input_shape()
    print("min shape:", min_shape)
    print("step input shape:", stepsize)
    shape = tuple(min_shape + 12 * stepsize)
    print("choose input shape:", shape)
    raw = tf.placeholder(tf.float32, shape=shape)
    raw_bc = tf.reshape(raw, (1, 1,) + shape)
    unet_out, fov, vx = model.unet(raw_bc)
    output, full_fov = ops3d.conv_pass(
        unet_out,
        kernel_size=[(1, 1, 1)],
        num_fmaps=1,
        activation=None,
        fov=fov,
        voxel_size=vx
        )

    tf.train.export_meta_graph(filename='unet.meta')

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        tf.summary.FileWriter('.', graph=tf.get_default_graph())

    print(output.shape)