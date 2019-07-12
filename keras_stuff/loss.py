import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import variables as tf_variables

from collections import defaultdict
import inspect
import numpy as np
import os

from keras.common import floatx
from keras.common import _EPSILON
from keras.common import image_data_format

# Legacy functions
from keras.common import set_image_dim_ordering
from keras.common import image_dim_ordering

# VARIABLE MANIPULATION

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)

def sparse_categorical_crossentropy(output, target, from_logits=False):
    """Categorical crossentropy with integer targets.
    # Arguments
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        target: An integer tensor.
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output)

    output_shape = output.get_shape()
    targets = cast(flatten(target), 'int64')
    logits = tf.reshape(output, [-1, int(output_shape[-1])])
    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets,
        logits=logits)
    if len(output_shape) == 3:
        # if our output includes timesteps we need to reshape
        return tf.reshape(res, tf.shape(output)[:-1])
    else:
        return res


def sparse_perfect_categorical_crossentropy(output, target, from_logits=False):
    """Categorical crossentropy with integer targets.
    # Arguments
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        target: An integer tensor.
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output)

    output_shape = output.get_shape()
    targets = cast(flatten(target), 'int64')
    logits = tf.reshape(output, [-1, int(output_shape[-1])])
    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets,
        logits=logits)
    if len(output_shape) == 3:
        # if our output includes timesteps we need to reshape
        return K.min(K.cast(K.equal(res, target), K.floatx()), axis=-1)
        #return tf.reshape(res, tf.shape(output)[:-1])
    else:
        return res

