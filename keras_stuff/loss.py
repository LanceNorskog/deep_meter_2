import tensorflow as tf

from collections import defaultdict
import inspect
import numpy as np
import os

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

def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy with integer targets.
    # Arguments
        target: An integer tensor.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.
    # Returns
        Output tensor.
    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # If the channels are not in the last axis, move them to be there:
    if axis != -1 and axis != output_dimensions[-1]:
        permutation = output_dimensions[:axis] + output_dimensions[axis + 1:]
        permutation += [axis]
        output = tf.transpose(output, perm=permutation)

    # Note: tf.nn.sparse_softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output)

    output_shape = output.get_shape()
    targets = cast(flatten(target), 'int64')
    logits = tf.reshape(output, [-1, tf.shape(output)[-1]])
    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets,
        logits=logits)
    if len(output_shape) >= 3:
        # if our output includes timestep dimension
        # or spatial dimensions we need to reshape
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

