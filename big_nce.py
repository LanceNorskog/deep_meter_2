# from https://github.com/joelthchao/keras branch with this item

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
import tensorflow as tf
import keras.backend as K
from keras import initializers,regularizers,constraints
from keras.models import Model
from keras.layers import Dense
from keras.engine.base_layer import InputSpec
from keras.engine.topology import Layer
from keras.engine.input_layer import Input

def nce_loss_function(kernel, bias, target, inputs, num_sampled, num_classes):
    return tf.nn.nce_loss(kernel, bias, target, inputs, num_sampled, num_classes)

class NCE(Layer):
    """Regular densely-connected NN layer with NCE Loss.
    `NCE` implements the operation:
    `output = dot(input, kernel) + bias`
    `kernel` is a weights matrix created by the layer, and `bias` is a bias vector
    created by the layer. Also, it adds NCE Loss to the model.
    See [reference](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf).
    # Example
    ```python
        inputs = Input(shape=(4,))
        target = Input(shape=(1,))  # sparse format, e.g. [1, 3, 2, 6, ...]
        net = Dense(8)(inputs)
        net = NCE(units=128, num_sampled=32)([net, target])
        model = Model(inputs=[inputs, target], outputs=net)
        model.compile(optimizer='adam', loss=None)
        x = np.random.rand(1000, 4)
        y = np.random.randint(128, size=1000)
        model.fit([x, y], None)
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space (num classes).
        num_sampled: Positive integer, number of classes to sample in NCE Loss.
        num_true: Max # of positive classes, pad to this for variable inputs
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        Two tensors. First one is 2D tensor with shape: `(batch_size, input_dim)`.
        Second one is 1D tensor with length `batch_size`
    # Output shape
        2D tensor with shape: `(batch_size, units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                 units,
                 num_sampled,
                 num_true=1,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NCE, self).__init__(**kwargs)
        self.units = units
        self.num_sampled = num_sampled
        if self.num_sampled > self.units:
            raise Exception('num_sample: {} cannot be greater than units: {}'.format(
                num_sampled, units))
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = [InputSpec(min_ndim=2), InputSpec(min_ndim=1)]
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        self.input_spec[0] = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        pred, target = inputs
        output = K.dot(pred, self.kernel)
        output = K.bias_add(output, self.bias, data_format='channels_last')

        nce_loss = nce_loss_function(
            K.transpose(self.kernel), self.bias, target, pred, self.num_sampled, self.units)
        self.add_loss(K.mean(nce_loss))
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[0][-1]
        output_shape = list(input_shape[0])
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'num_sampled': self.num_sampled,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(NCE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    inputs = Input(shape=(4,))
    target = Input(shape=(1,))  # sparse format, e.g. [1, 3, 2, 6, ...]
    net = Dense(8)(inputs)
    net = NCE(units=128, num_sampled=32)([net, target])
    model = Model(inputs=[inputs, target], outputs=net)
    model.compile(optimizer='adam', loss=None)
    x = np.random.rand(1000, 4)
    y = np.random.randint(128, size=1000)
    history = model.fit([x, y], None)
    for key in history.history.keys():
        print(key)
    print(len(y))
    print(y)
