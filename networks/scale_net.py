import tensorflow as tf
import numpy as np
from . import ops3d
import warnings


class ScaleNet(object):
    def __init__(self, list_of_serialunets, input_shape, name="scnet"):
        """
        :param list list_of_serialunets: list of instances of SerialUNet, sorted by increasing voxel size of the input
        :param tuple input_shape: shape of input tensor in (z, y, x)
        """
        self.name = name
        # highest to lowest resolution
        self.list_of_serialunets = list_of_serialunets
        self.input_shapes = [input_shape]
        bottom_shape = self.list_of_serialunets[0].get_bottom_shape_from_input_shape(
            input_shape
        )
        output_shape = self.list_of_serialunets[0].get_output_shape_from_input_shape(
            input_shape
        )
        self.bottom_shapes = [bottom_shape]
        self.output_shapes = [output_shape]
        self.padding_orig_vx = [(np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)))]
        for serialunet in self.list_of_serialunets[1::]:
            output_shape = np.copy(bottom_shape)

            # assert (output_shape - serialunet.min_output_shape)%serialunet.step_valid_shape ==0
            if (output_shape < serialunet.min_output_shape).any():
                too_small_dim = output_shape < serialunet.min_output_shape
                output_shape[too_small_dim] = serialunet.min_output_shape[too_small_dim]

            if (
                (output_shape - serialunet.min_output_shape)
                % serialunet.step_valid_shape
                != 0
            ).any():
                output_shape += (
                    serialunet.step_valid_shape
                    - (output_shape - serialunet.min_output_shape)
                    % serialunet.step_valid_shape
                )

            input_shape = serialunet.get_input_shape_from_output_shape(output_shape)
            bottom_shape = serialunet.get_bottom_shape_from_input_shape(input_shape)
            self.input_shapes.append(input_shape)
            self.output_shapes.append(output_shape)
            self.bottom_shapes.append(bottom_shape)
        print("input_shapes before: ", self.input_shapes)
        print("bottom_shapes before: ", self.bottom_shapes)
        print("output_shapes before: ", self.output_shapes)
        for k in range(len(self.list_of_serialunets))[:0:-1]:
            print("unet", k)

            if ((self.output_shapes[k] - self.bottom_shapes[k - 1]) % 2 != 0).any():
                print("changing things")
                odd_dim = (self.output_shapes[k] - self.bottom_shapes[k - 1]) % 2 != 0
                self.bottom_shapes[k - 1][odd_dim] += 1
                self.output_shapes[k - 1][odd_dim] += self.list_of_serialunets[
                    k - 1
                ].step_valid_shape[odd_dim]
                self.input_shapes[k - 1][odd_dim] += self.list_of_serialunets[
                    k - 1
                ].step_valid_shape[odd_dim]
        print("input_shapes after: ", self.input_shapes)
        print("bottom_shapes after: ", self.bottom_shapes)
        print("output_shapes after: ", self.output_shapes)

        self.voxel_sizes = [np.array(list_of_serialunets[0].input_voxel_size)]
        for serialunet in list_of_serialunets[:-1]:
            self.voxel_sizes.append(self.voxel_sizes[-1] * serialunet.step_valid_shape)

        for lv in range(len(self.list_of_serialunets))[1:]:
            # padding_left = np.copy(self.padding_orig_vx[-1][0])
            # padding_right = np.copy(self.padding_orig_vx[-1][1])
            padding_left = -self.list_of_serialunets[lv - 1].get_downward_padding() / 2
            padding_right = np.copy(padding_left)

            padding_left += (self.voxel_sizes[lv] / self.voxel_sizes[lv - 1]) * (
                (self.output_shapes[lv] - self.bottom_shapes[lv - 1]) // 2
            )
            padding_right += (self.voxel_sizes[lv] / self.voxel_sizes[lv - 1]) * (
                (
                    self.output_shapes[lv]
                    - self.bottom_shapes[lv - 1]
                    + np.array((1, 1, 1))
                )
                // 2
            )
            padding_left += (
                (self.voxel_sizes[lv] / self.voxel_sizes[lv - 1])
                * (self.input_shapes[lv] - self.output_shapes[lv])
                / 2
            )
            padding_right += (
                (self.voxel_sizes[lv] / self.voxel_sizes[lv - 1])
                * (self.input_shapes[lv] - self.output_shapes[lv])
                / 2
            )
            padding_left *= self.voxel_sizes[lv - 1]
            padding_left += self.padding_orig_vx[-1][0]
            padding_right *= self.voxel_sizes[lv - 1]
            padding_right += self.padding_orig_vx[-1][1]
            self.padding_orig_vx.append((padding_left, padding_right))

    def build(self, inputs):
        bottom_insert = None
        for serialunet, inp, vox in zip(
            self.list_of_serialunets[::-1], inputs[::-1], self.voxel_sizes[::-1]
        ):
            print("Building Unet with voxelsize {0:}".format(vox))
            bottom_insert, fov, vs = serialunet.build(
                inp,
                fmaps_bottom=bottom_insert,
                scope="unet_at_{0:}-{1:}-{2:}".format(vox[0], vox[1], vox[2]),
            )
        return bottom_insert, fov, vs


class SerialUNet(object):
    def __init__(
        self,
        num_fmaps_down,
        num_fmaps_up,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        activation="relu",
        input_fov=(1, 1, 1),
        input_voxel_size=(1, 1, 1),
        constant_upsample=False,
    ):
        """
        :param list num_fmaps_down: number of feature maps on the downward path
        :param list num_fmaps_up: number of feature maps on the upward path
        :param tuple/list fmap_inc_factors: multiplication factors for number of feature maps, can be different per
                                            level
        :param tuple/list downsample_factors: list or tuple of factors (given as tuple (z, y, x)) by which feature maps
                                              are downsampled going from level to level
        :param tuple/list kernel_size_down: list or tuple for kernel sizes to be used on the downsampling path of the
                                            U-Net, sorted by increasing voxel size. Each element is a list/tuple of
                                            kernel sizes (given as tuple (z, y, x)) to be successively applied on one
                                            level (i.e. length determines the number of convolutions per level)
        :param tuple/list kernel_size_up: list or tuple for kernel sizes to be used on the upsampling path of the
                                          U-Net, sorted by increasing voxel size. Each element is a list/tuple of
                                          kernel sizes (given as tuple (z, y, x)) to be successively applied on one
                                          level (i.e. length determines the number of convolutions per level). The
                                          bottom level is considered path of the downsampling path (entries in
                                          kernel_size_up will be ignored)
        :param string activation: activation function used after each convolutional layer
        :param tuple input_fov: field of view of the input given as tuple (z, y, x)
        :param tuple input_voxel_size: voxel size of the input given as tuple (z, y, x)
        """
        # if isinstance(fmap_inc_factors, int):
        #    fmap_inc_factors = [fmap_inc_factors] * len(downsample_factors)
        assert (
            len(num_fmaps_down) - 1
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
        self.input_voxel_size = input_voxel_size
        self.constant_upsample = constant_upsample
        self.min_input_shape, self.step_valid_shape, self.min_output_shape, self.min_bottom_shape = (
            self.compute_minimal_shapes()
        )

    def compute_minimal_shapes(self):
        # compute minimal shape in the bottom layer (after the convolutions s.t. the upward path can still be evaluated
        min_bottom_right = (1.0, 1.0, 1.0)
        for lv in range(len(self.downsample_factors)):

            kernels = np.copy(self.kernel_size_up[lv])

            conv_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )  # padding needed for
            # convolutions on upsamling side on level lv

            min_bottom_right += conv_pad / np.prod(self.downsample_factors[lv:], axis=0)

        min_bottom_right = np.ceil(min_bottom_right)
        min_bottom_right = np.max([min_bottom_right, (1.0, 1.0, 1.0)], axis=0)
        min_input_shape = np.copy(min_bottom_right)

        for lv in range(len(self.kernel_size_down))[::-1]:

            if lv != len(self.kernel_size_down) - 1:
                min_input_shape *= self.downsample_factors[lv]

            kernels = np.copy(self.kernel_size_down[lv])
            conv_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )
            min_input_shape += conv_pad

            if lv == len(self.kernel_size_down) - 1:
                min_bottom_left = np.copy(min_input_shape)

        min_output_shape = np.copy(min_bottom_right)
        for lv in range(len(self.downsample_factors))[::-1]:
            min_output_shape *= self.downsample_factors[lv]
            kernels = np.copy(self.kernel_size_up[lv])
            conv_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )
            min_output_shape -= conv_pad

        step = np.prod(self.downsample_factors, axis=0)

        return min_input_shape, step, min_output_shape, min_bottom_left

    def get_downward_padding(self):
        # padding here means the difference between the complete shapes, not just one side
        return self.min_input_shape - self.min_bottom_shape * self.step_valid_shape

    def is_valid_input_shape(self, input_shape):
        if (input_shape >= self.min_input_shape).all() and (
            (input_shape - self.min_input_shape) % self.step_valid_shape == 0
        ).all():
            return True
        else:
            return False

    def get_bottom_shape_from_input_shape(self, input_shape):
        if not self.is_valid_input_shape(input_shape):
            raise ValueError("{0:} is not a valid input shape".format(input_shape))
        bottom_shape = np.copy(input_shape)
        for lv in range(len(self.downsample_factors)):
            kernels = np.copy(self.kernel_size_down[lv])
            conv_pad = np.sum(
                [np.array(k) - np.array((1.0, 1.0, 1.0)) for k in kernels], axis=0
            )
            bottom_shape -= conv_pad
            bottom_shape /= self.downsample_factors[lv]
        return bottom_shape

    def get_input_shape_from_output_shape(self, output_shape):
        input_shape = output_shape + self.min_input_shape - self.min_output_shape
        if not self.is_valid_input_shape(input_shape):
            raise ValueError("{0:} is not a valid output shape".format(output_shape))

        return input_shape

    def get_output_shape_from_input_shape(self, input_shape):
        if not self.is_valid_input_shape(input_shape):
            raise ValueError("{0:} is not a valid input shape".format(input_shape))
        output_shape = input_shape - self.min_input_shape + self.min_output_shape
        return output_shape

    def build(
        self,
        fmaps_in,
        fmaps_bottom=None,
        num_fmaps_down=None,
        num_fmaps_up=None,
        downsample_factors=None,
        kernel_size_down=None,
        kernel_size_up=None,
        activation=None,
        layer=0,
        fov=None,
        voxel_size=None,
        scope="",
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
        print(prefix + "Creating U-Net layer %i" % layer)
        print(prefix + "f_in: " + str(fmaps_in.shape))

        # convolve
        with tf.variable_scope(scope + "lev%i" % layer):
            bottom_layer = layer == len(downsample_factors)

            # concatenate second input feature map at bottom layer
            if bottom_layer and fmaps_bottom is not None:
                assert (
                    (
                        np.array(fmaps_in.get_shape().as_list())
                        - np.array(fmaps_bottom.get_shape().as_list())
                    )
                    % 2
                    == 0
                ).all()
                fmaps_bottom_cropped = ops3d.crop_zyx(
                    fmaps_bottom, fmaps_in.get_shape().as_list()
                )
                fmaps_in = tf.concat([fmaps_bottom_cropped, fmaps_in], 1)

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
            if bottom_layer:
                print(prefix + "bottom layer")
                print(prefix + "f_out: " + str(f_left.shape))
                return f_left, fov, voxel_size

            # downsample

            g_in, fov, voxel_size = ops3d.downsample(
                f_left,
                downsample_factors[layer],
                name='unet_down_%i_to_%i' % (layer, layer + 1),
                fov=fov,
                voxel_size=voxel_size,
                prefix=prefix,
            )

            # recursive U-net
            g_out, fov, voxel_size = self.build(
                g_in,
                fmaps_bottom=fmaps_bottom,
                num_fmaps_down=num_fmaps_down,
                num_fmaps_up=num_fmaps_up,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                activation=activation,
                layer=layer + 1,
                fov=fov,
                voxel_size=voxel_size,
                scope=scope,
            )

            print(prefix + "g_out: " + str(g_out.shape))

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

            print(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

            # copy-crop
            f_left_cropped = ops3d.crop_zyx(
                f_left, g_out_upsampled.get_shape().as_list()
            )

            print(prefix + "f_left_cropped: " + str(f_left_cropped.shape))

            # concatenate along channel dimension
            f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)

            print(prefix + "f_right: " + str(f_right.shape))

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

            print(prefix + "f_out: " + str(f_out.shape))

        return f_out, fov, voxel_size


if __name__ == "__main__":
    unet0 = SerialUNet(
        36,
        2,
        [(1, 3, 3), (1, 3, 3)],
        [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
        [[(1, 3, 3), (1, 3, 3)], [(1, 3, 3), (1, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
    )
    unet1 = SerialUNet(
        72,
        2,
        [(3, 3, 3), (3, 3, 3)],
        [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
        [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
    )
    print(
        unet0.min_input_shape,
        unet0.min_output_shape,
        unet0.min_bottom_shape,
        unet0.step_valid_shape,
    )
    print(
        unet1.min_input_shape,
        unet1.min_output_shape,
        unet1.min_bottom_shape,
        unet1.step_valid_shape,
    )
    comb = ScaleNet([unet0, unet1], unet0.min_input_shape)
    print(comb.padding_orig_vx)
    raw = tf.placeholder(tf.float32, shape=unet0.min_input_shape)
    raw_batched = tf.reshape(raw, (1, 1) + tuple(unet0.min_input_shape.astype(np.int)))
    inputs = [raw_batched]
    for inp in comb.input_shapes[1:]:
        raw_downscaled = tf.placeholder(tf.float32, shape=inp)
        raw_downscaled_batched = tf.reshape(
            raw_downscaled, (1, 1) + tuple(inp.astype(np.int))
        )
        inputs.append(raw_downscaled_batched)
    comb.build(inputs)

    tf.train.export_meta_graph(filename="combnet.meta")

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        tf.summary.FileWriter(".", graph=tf.get_default_graph())
