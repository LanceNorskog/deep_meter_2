import tensorflow as tf
import numpy as np
from keras.models import Model
from keras import backend as K
from keras import initializers,regularizers,constraints
from keras.models import Model
from keras.layers import Dense
from keras.engine.base_layer import InputSpec
from keras.engine.topology import Layer
from keras.engine.input_layer import Input

np.random.seed(10)

import random


# from https://github.com/joelthchao/keras branch with this item
# Sampling output layer, various samplers

def nce_loss_function(weights, biases, labels, inputs, num_sampled, num_classes, num_true):
    print("labels {0}, inputs {1}".format(str(labels.shape), str(inputs.shape)))
    if K.learning_phase() == 1:
        loss = tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true,
            partition_strategy="div")
    else:
        logits = tf.matmul(inputs, tf.transpose(weights))
        logits = tf.nn.bias_add(logits, biases)
        labels_one_hot = tf.one_hot(labels, num_classes)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            #labels=labels_one_hot,
            labels=labels_one_hot[:][0][:],
            logits=logits)
        loss = tf.reduce_sum(loss, axis=1)
    return loss

def sampled_softmax_loss_function(weights, biases, labels, inputs, num_sampled, num_classes, num_true):
    print("labels {0}, inputs {1}".format(str(labels.shape), str(inputs.shape)))
    if K.learning_phase() == 1:
        return tf.nn.sampled_softmax_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true, 
            partition_strategy="div")
    else:
        logits = tf.matmul(inputs, tf.transpose(weights))
        logits = tf.nn.bias_add(logits, biases)
        labels_one_hot = tf.one_hot(labels, num_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            #labels=labels_one_hot,
            labels=labels_one_hot[:][0][:],
            logits=logits)
        return loss

class Sampling(Layer):
    """Regular densely-connected NN layer with various sampling Loss.
    `Sampling` implements the operation:
    `output = dot(input, kernel) + bias`
    `kernel` is a weights matrix created by the layer, and `bias` is a bias vector
    created by the layer. Also, it adds a sampling Loss to the model.
    See [reference](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf).
    # Example
    ```python
        inputs = Input(shape=(4,))
        target = Input(shape=(1,))  # sparse format, e.g. [1, 3, 2, 6, ...]
        net = Dense(8)(inputs)
        net = Sampling(units=128, num_sampled=32)([net, target])
        model = Model(inputs=[inputs, target], outputs=net)
        model.compile(optimizer='adam', loss=None)
        x = np.random.rand(1000, 4)
        y = np.random.randint(128, size=1000)
        model.fit([x, y], None)
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space (num classes).
        num_sampled: Positive integer, number of classes to sample in Sampling Loss.
        type: 'sampled_softmax', 'nce'
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
                 type='sampled_softmax',
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
        super(Sampling, self).__init__(**kwargs)
        self.units = units
        self.num_sampled = num_sampled
        if self.num_sampled > self.units:
            raise Exception('num_sample: {} cannot be greater than units: {}'.format(
                num_sampled, units))
        self.type = type
        if not (self.type == 'nce' or self.type == 'sampled_softmax'):
            raise Exception('type {} is not a valid sampling loss type'.format(type))
        self.num_true = num_true
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

        # TODO : check train or test mode
        if self.type == 'nce':
            nce_loss = nce_loss_function(
                K.transpose(self.kernel), self.bias, target, pred, self.num_sampled, self.units, self.num_true)
            self.add_loss(K.mean(nce_loss))
        else:
            sampled_softmax_loss = sampled_softmax_loss_function(
                K.transpose(self.kernel), self.bias, target, pred, self.num_sampled, self.units, self.num_true)
            self.add_loss(K.mean(sampled_softmax_loss))
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
        base_config = super(Sampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Zipfian test suite
# number of test samples
num_train = 32*500
num_test = 32
num_valid = 100
num_epochs = 2
num_hidden = 10
# number of classes
input_width = 2000
# number of classes
output_width = 800
# number of samples for NCE
num_sampled = 24
# number of labels
num_true = 1

def fill_zipf(length, input_width, output_width, num_true=1):
  input_onehot = np.zeros((length, input_width), dtype='float32')
  output_labels = np.zeros((length, num_true), dtype='int32')
  rand = np.random.zipf(1.2, length * num_true) % input_width
  for i in range(length):
    for t in range(num_true):
      k = rand[t * length + i]
      input_onehot[i][k] = 1.0
      output_labels[i][t] = min(output_width - 1, int(k * output_width/input_width))
  return input_onehot, output_labels

# choose one of the two
sampling_type='sampled_softmax'
sampling_type='nce'
inputs = Input(shape=(input_width,))
target = Input(shape=(num_true,), dtype=tf.int64)  
net = Dense(input_width)(inputs)
net = Dense(num_hidden, activation='relu')(net)
net = Sampling(units=output_width, num_sampled=num_sampled, type=sampling_type, num_true=num_true)([net, target])
model = Model(inputs=[inputs, target], outputs=net)
model.compile(optimizer='adam', loss=None, metrics=['binary_crossentropy'])
model.summary()

train_onehot, train_labels = fill_zipf(num_train, input_width, output_width, num_true)
model.fit([train_onehot, train_labels], None, 
    epochs=num_epochs, verbose=2)

test_input, test_output = fill_zipf(num_test, input_width, output_width, num_true)
predicts = model.predict([test_input, test_output], batch_size=32)
count = 0
for test in range(num_test):
  pred = predicts[test]
  topindexes = list(np.argsort(pred))[(output_width-num_true):]
  for t in range(num_true):
    if test_output[test][t] in topindexes:
      count += 1
print("Average of {0} correct labels out of {1} tests".format(count/num_true, num_test))

