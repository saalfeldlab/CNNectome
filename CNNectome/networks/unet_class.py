import tensorflow as tf
import numpy as np
from . import ops3d
import warnings
import logging
import typing

int_dim_expandable = typing.Union[int, typing.Sequence[int]]
bool_dim_expandable = typing.Union[bool, typing.Sequence[bool]]


class UNet(object):

    def __init__(
        self,
        num_fmaps_down: typing.Sequence[int],
        num_fmaps_up: typing.Sequence[int],
        downsample_factors: typing.Sequence[typing.Sequence[int]],
        kernel_size_down: typing.Sequence[typing.Sequence[int_dim_expandable]],
        kernel_size_up: typing.Sequence[typing.Sequence[int_dim_expandable]],
        skip_connections: bool_dim_expandable = True,
        activation: str = "relu",
        padding: str = "valid",
        constant_upsample: bool = True,
        trans_equivariant: bool = True,
        enforce_even_context: bool = False,
        input_fov: typing.Tuple[int, int, int] = (1, 1, 1),
        input_voxel_size: typing.Tuple[int, int, int] = (1, 1, 1),
    ) -> None:
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
        if not isinstance(skip_connections, (list, tuple)):
            if skip_connections is True:
                self.skip_connections = [True, ] * len(self.downsample_factors)
            elif skip_connections is False:
                self.skip_connections = [False, ] * len(self.downsample_factors)
            else:
                raise ValueError("Can't handle input for skip connections: {0:}".format(skip_connections))
        else:
            self.skip_connections = skip_connections
        self.activation = activation
        self.input_fov = input_fov
        self.padding = padding
        self.constant_upsample = constant_upsample
        self.trans_equivariant = trans_equivariant
        self.enforce_even_context = enforce_even_context
        self.input_voxel_size = input_voxel_size
        self.min_input_shape, self.step_valid_shape, self.min_output_shape, self.min_bottom_shape = (
            self.compute_minimal_shapes()
        )

    def compute_minimal_shapes(self) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the minimal input shape, shape at the bootleneck and output shape as well as suitable step sizes
        (additional context) for the given U-Net configuration. This is computed for U-Nets with `valid` padding as
        well as for U-Nets with `same` padding. For `same` padding U-Nets these requirements are not strict, but
        represent the minimum shape for which voxels that are seeing a full field of view are contained in the output
        and thus making it easy to switch to a `valid` padding U-Net for inference

        Returns:
            A 4-element tuple containing, respectively, the minimum input shape and valid step size, the corresponding
            minimum output shape and minimum bottleneck shape, i.e. shape after last downsampling.
        """

        # valid step (meaning what values can be added on top of the minimum shape to also produce a U-Net with valid
        # shapes
        step = np.prod(self.downsample_factors, axis=0)

        # PART 1: calculate the minimum shape of the feature map after convolutions in the bottom layer ("bottom
        # right") such that a feature map size of 1 can be guaranteed throughout the upsampling paths

        # initialize with a minimum shape of 1 (representing the size after convolutions in each level)
        min_bottom_right = [(1.0, 1.0, 1.0)] * (len(self.downsample_factors) + 1)

        # propagate those minimal shapes back through the network to calculate the corresponding minimal shapes on the
        # "bottom right"

        # iterate over levels of unet
        for lv in range(len(self.downsample_factors)):
            kernels = np.copy(self.kernel_size_up[lv])

            # padding added by convolution kernels on current level (upsampling path)
            total_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )

            if self.enforce_even_context:
                assert np.all(total_pad % 2 == 0), \
                    "Kernels {kernels:} on level {lv:} of U-Net (upsampling path) not compatible with enforcing an " \
                    "even context".format(kernels=kernels, lv=lv)

            # for translational equivariance U-Net includes cropping to the stride of the downsampling factors
            if self.trans_equivariant:
                # rounding up the padding to the closest multiple of what is the crop factor because the result of the
                # upsampling will be a multiple of the crop factor, and crop_to_factor makes it such that after the
                # operations on this level the feature map will also be a multiple of the crop factor, i.e. the
                # total_pad needs to be a multiple of the crop factor as well

                total_pad = np.ceil(
                    total_pad
                    / np.prod(self.downsample_factors[lv:], axis=0, dtype=np.float)
                ) * np.prod(self.downsample_factors[lv:], axis=0)

                # when even context are enforced the padding needs to be even so trans_equivariant will crop +1
                # factors if the otherwise resulting padding is odd
                if self.enforce_even_context:
                    total_pad += (total_pad%2)*np.prod(self.downsample_factors[lv:], axis=0)

            for l in range(lv + 1):
                min_bottom_right[l] += total_pad # add the padding added by convolution
                min_bottom_right[l] /= self.downsample_factors[lv] # divide by downsampling factor of current level

        # round up the fractions potentially introduced by downsampling factor division
        min_bottom_right = np.ceil(min_bottom_right)

        # take the max across levels (i.e. we find the level that required the most context)
        min_bottom_right = np.max(min_bottom_right, axis=0)

        # PART 2: calculate the minimum input shape by propagating from the "bottom right" to the input of the U-Net
        min_input_shape = np.copy(min_bottom_right)

        for lv in range(len(self.kernel_size_down))[::-1]:  # go backwards through downsampling path

            if lv != len(self.kernel_size_down) - 1:  # unless bottom layer
                min_input_shape *= self.downsample_factors[lv]  # calculate shape before downsampling

            # calculate shape before convolutions on current level
            kernels = np.copy(self.kernel_size_down[lv])
            total_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )
            if self.enforce_even_context:
                assert np.all(total_pad % 2 == 0), \
                    "Kernels {kernels:} on level {lv:} of U-Net (downsampling path) not compatible with enforcing an " \
                    "even context".format(kernels=kernels, lv=lv)

            min_input_shape += total_pad

            # side product: shape before convolutions on bottom level
            if lv == len(self.kernel_size_down) - 1:
                min_bottom_left = np.copy(min_input_shape)

        # PART 3: calculate the minimum output shape by propagating from the "bottom right" to the output of the U-Net
        min_output_shape = np.copy(min_bottom_right)
        for lv in range(len(self.downsample_factors))[::-1]: # go through upsampling path
            min_output_shape *= self.downsample_factors[lv] # calculate shape after upsampling

            # calculate shape after convolutions on current level
            kernels = np.copy(self.kernel_size_up[lv])
            total_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )

            # same rational for translational equivariance as above in PART 1
            if self.trans_equivariant:
                total_pad = np.ceil(
                    total_pad
                    / np.prod(self.downsample_factors[lv:], axis=0, dtype=np.float)
                ) * np.prod(self.downsample_factors[lv:], axis=0)
            min_output_shape -= total_pad

        return min_input_shape, step, min_output_shape, min_bottom_left

    def build(
        self,
        fmaps_in: tf.Tensor,
        num_fmaps_down: typing.Optional[typing.Sequence[int]] = None,
        num_fmaps_up: typing.Optional[typing.Sequence[int]] = None,
        downsample_factors: typing.Optional[typing.Sequence[int_dim_expandable]] = None,
        kernel_size_down: typing.Optional[typing.Sequence[int_dim_expandable]] = None,
        kernel_size_up: typing.Optional[typing.Sequence[int_dim_expandable]] = None,
        skip_connections: typing.Optional[bool_dim_expandable] = None,
        activation: typing.Optional[str] = None,
        padding: typing.Optional[str] = None,
        layer: int = 0,
        fov: typing.Optional[typing.Tuple[int, int, int]] = None,
        voxel_size: typing.Optional[typing.Tuple[int, int, int]] = None,
    ) -> typing.Tuple[tf.Tensor, typing.Tuple[int, int, int], typing.Tuple[int, int, int]]:

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
        if skip_connections is None:
            skip_connections = self.skip_connections
        if activation is None:
            activation = self.activation
        if padding is None:
            padding = self.padding
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
                padding=padding,
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
            logging.info(prefix + "after convs: " + str(f_left.shape))
            # downsample

            g_in, fov, voxel_size = ops3d.downsample(
                f_left,
                downsample_factors[layer],
                padding=padding,
                name="unet_down_%i_to_%i" % (layer, layer + 1),
                fov=fov,
                voxel_size=voxel_size,
                prefix=prefix,
            )
            logging.info(prefix + "after downsample:" + str(g_in.shape))

            # recursive U-net
            g_out, fov, voxel_size = self.build(
                g_in,
                num_fmaps_down=num_fmaps_down,
                num_fmaps_up=num_fmaps_up,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                skip_connections=skip_connections,
                padding=padding,
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
                padding=padding,
                name="unet_up_%i_to_%i" % (layer + 1, layer),
                fov=fov,
                voxel_size=voxel_size,
                prefix=prefix,
                constant_upsample=self.constant_upsample,
            )

            logging.info(prefix + "after upsample: " + str(g_out_upsampled.shape))

            if padding == "valid":
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
                        enforce_even_context=self.enforce_even_context
                    )
                    logging.info(prefix + "after crop_to_factor: " + str(g_out_upsampled.shape))
                if skip_connections[layer]: # can skip this step if there's no skip conneciton here
                    # copy-crop
                    f_left = ops3d.crop_zyx(
                        f_left, g_out_upsampled.get_shape().as_list(), enforce_even_context=self.enforce_even_context
                    )
                    logging.info(prefix + "f_left_cropped: " + str(f_left.shape))
            else:
                if f_left.get_shape() != g_out_upsampled.get_shape():
                    g_out_upsampled = ops3d.crop_zyx(
                        g_out_upsampled, f_left.get_shape().as_list(), enforce_even_context=False
                    )
                    logging.info(prefix + "g_out_upsampled_cropped: " + str(g_out_upsampled.shape))
            if skip_connections[layer]:
                f_right = tf.concat([f_left, g_out_upsampled], 1)
            else:
                f_right = g_out_upsampled

            logging.info(prefix + "after concat: " + str(f_right.shape))

            # convolve
            f_out, fov = ops3d.conv_pass(
                f_right,
                kernel_size=kernel_size_up[layer],
                num_fmaps=num_fmaps_up[layer],
                padding=padding,
                name="unet_layer_%i_right" % layer,
                fov=fov,
                voxel_size=voxel_size,
                prefix=prefix,
            )

            logging.info(prefix + "after conv: " + str(f_out.shape))

        return f_out, fov, voxel_size
