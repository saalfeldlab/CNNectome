from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
import tensorflow as tf


def ignore(x, binary_tensor, name=None):
    with ops.name_scope(name, "ignore", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        keep_ratio = math_ops.divide(
            math_ops.reduce_sum(binary_tensor),
            math_ops.reduce_prod(
                array_ops.shape(binary_tensor, out_type=dtypes.float32)
            ),
        )
        keep_ratio.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        with tf.Session() as sess:

            print(keep_ratio.eval(session=sess))
        ret = math_ops.div(x, keep_ratio) * binary_tensor
        ret.set_shape(x.get_shape())
        return ret


def tf_var_summary(var):
    # compute mean of variable
    mean = tf.reduce_mean(var)
    tf.summary.scalar("mean_" + var.name, mean)

    # compute std of variable
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

    tf.summary.scalar("stddev_" + var.name, stddev)
    tf.summary.scalar("max_" + var.name, tf.reduce_max(var))
    tf.summary.scalar("min_" + var.name, tf.reduce_min(var))
    tf.summary.histogram("histogram_" + var.name, var)
