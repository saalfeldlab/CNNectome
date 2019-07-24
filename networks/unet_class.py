
import tensorflow as tf
import numpy as np
from . import ops3d
import warnings
import logging


class UNet(object):
    def __init__(
        self,
        num_fmaps_down,
        num_fmaps_up,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        activation="relu",
        constant_upsample=False,
        trans_equivariant=False,
        input_fov=(1, 1, 1),
        input_voxel_size=(1, 1, 1),
    ):
        assert (
            len(num_fmaps_down) - 1
            == len(num_fmaps_up) - 1
            == len(downsample_factors)
            == len(kernel_size_down) - 1
        )
        assert len(downsample_factors) == len(kernel_size_up) - 1 or len(
            downsample_factors
        ) == len(kernel_size_up)

        if len(downsample_factors) == len(kernel_size_up) - 1:
            warnings.warn(
                "kernel sizes for upscaling are not used for the bottom layer"
            )
            kernel_size_up = kernel_size_up[:-1]
        self.num_fmaps_down = num_fmaps_down
        self.num_fmaps_up = num_fmaps_up
        self.downsample_factors = downsample_factors
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up = kernel_size_up
        self.activation = activation
        self.input_fov = input_fov
        self.constant_upsample = constant_upsample
        self.trans_equivariant = trans_equivariant
        self.input_voxel_size = input_voxel_size
        self.min_input_shape, self.step_valid_shape, self.min_output_shape, self.min_bottom_shape = (
            self.compute_minimal_shapes()
        )

    def compute_minimal_shapes(self):
        # compute minimal shape in the bottom layer (after the convolutions s.t. the upward path can still be evaluated
        min_bottom_right = [(1.0, 1.0, 1.0)] * (len(self.downsample_factors) + 1)
        for lv in range(len(self.downsample_factors)):
            kernels = np.copy(self.kernel_size_up[lv])

            total_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )  # padding needed for
            # convolutions on upsamling side on level lv
            if self.trans_equivariant:
                total_pad = np.ceil(
                    total_pad
                    / np.prod(self.downsample_factors[lv:], axis=0, dtype=np.float)
                ) * np.prod(self.downsample_factors[lv:], axis=0)

            for l in range(lv + 1):
                min_bottom_right[l] += total_pad
                min_bottom_right[l] /= self.downsample_factors[lv]

        min_bottom_right = np.ceil(min_bottom_right)
        min_bottom_right = np.max(min_bottom_right, axis=0)
        min_input_shape = np.copy(min_bottom_right)

        for lv in range(len(self.kernel_size_down))[::-1]:

            if lv != len(self.kernel_size_down) - 1:
                min_input_shape *= self.downsample_factors[lv]

            kernels = np.copy(self.kernel_size_down[lv])
            total_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )
            min_input_shape += total_pad

            if lv == len(self.kernel_size_down) - 1:
                min_bottom_left = np.copy(min_input_shape)

        min_output_shape = np.copy(min_bottom_right)
        for lv in range(len(self.downsample_factors))[::-1]:
            min_output_shape *= self.downsample_factors[lv]
            kernels = np.copy(self.kernel_size_up[lv])
            total_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )
            if self.trans_equivariant:
                total_pad = np.ceil(
                    total_pad
                    / np.prod(self.downsample_factors[lv:], axis=0, dtype=np.float)
                ) * np.prod(self.downsample_factors[lv:], axis=0)

            min_output_shape -= total_pad

        step = np.prod(self.downsample_factors, axis=0)
        return min_input_shape, step, min_output_shape, min_bottom_left

    def build(
        self,
        fmaps_in,
        num_fmaps_down=None,
        num_fmaps_up=None,
        downsample_factors=None,
        kernel_size_down=None,
        kernel_size_up=None,
        activation=None,
        layer=0,
        fov=None,
        voxel_size=None,
    ):

        """Create a U-Net::
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

        """
        if num_fmaps_down is None:
            num_fmaps_down = self.num_fmaps_down
        if num_fmaps_up is None:
            num_fmaps_up = self.num_fmaps_up
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
        logging.info(prefix + "Creating U-Net layer %i" % layer)
        logging.info(prefix + "f_in: " + str(fmaps_in.shape))

        # convolve
        with tf.name_scope("lev%i" % layer):

            f_left, fov = ops3d.conv_pass(
                fmaps_in,
                kernel_size=kernel_size_down[layer],
                num_fmaps=num_fmaps_down[layer],
                activation=activation,
                name="unet_layer_%i_left" % layer,
                fov=fov,
                voxel_size=voxel_size,
                prefix=prefix,
            )

            # last layer does not recurse
            bottom_layer = layer == len(downsample_factors)

            if bottom_layer:
                logging.info(prefix + "bottom layer")
                logging.info(prefix + "f_out: " + str(f_left.shape))
                return f_left, fov, voxel_size

            # downsample

            g_in, fov, voxel_size = ops3d.downsample(
                f_left,
                downsample_factors[layer],
                name="unet_down_%i_to_%i" % (layer, layer + 1),
                fov=fov,
                voxel_size=voxel_size,
                prefix=prefix,
            )

            # recursive U-net
            g_out, fov, voxel_size = self.build(
                g_in,
                num_fmaps_down=num_fmaps_down,
                num_fmaps_up=num_fmaps_up,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                activation=activation,
                layer=layer + 1,
                fov=fov,
                voxel_size=voxel_size,
            )

            logging.info(prefix + "g_out: " + str(g_out.shape))

            # upsample
            g_out_upsampled, voxel_size = ops3d.upsample(
                g_out,
                downsample_factors[layer],
                num_fmaps_up[layer],
                activation=activation,
                name="unet_up_%i_to_%i" % (layer + 1, layer),
                fov=fov,
                voxel_size=voxel_size,
                prefix=prefix,
                constant_upsample=self.constant_upsample,
            )

            logging.info(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

            # crop for translation equivariance
            if self.trans_equivariant:
                factor_product = None
                for factor in downsample_factors[layer:]:
                    if factor_product is None:
                        factor_product = list(factor)
                    else:
                        factor_product = list(
                            f * ff for f, ff in list(zip(factor, factor_product))
                        )
                g_out_upsampled = ops3d.crop_to_factor(
                    g_out_upsampled,
                    factor=factor_product,
                    kernel_sizes=kernel_size_up[layer],
                )

            # copy-crop
            f_left_cropped = ops3d.crop_zyx(
                f_left, g_out_upsampled.get_shape().as_list()
            )
            logging.info(prefix + "f_left_cropped: " + str(f_left_cropped.shape))

            # concatenate along channel dimension
            f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)

            logging.info(prefix + "f_right: " + str(f_right.shape))

            # convolve
            f_out, fov = ops3d.conv_pass(
                f_right,
                kernel_size=kernel_size_up[layer],
                num_fmaps=num_fmaps_up[layer],
                name="unet_layer_%i_right" % layer,
                fov=fov,
                voxel_size=voxel_size,
                prefix=prefix,
            )

            logging.info(prefix + "f_out: " + str(f_out.shape))

        return f_out, fov, voxel_size
