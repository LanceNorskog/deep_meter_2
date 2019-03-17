# Manage Transfer Learning models

# boilerplate from base notebook
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import keras.layers as layers
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam, Adam
import gc
from google.colab import files
from google.colab import drive

import pickle
np.random.seed(10)

def define_embedding_function(embed):
    def UniversalEmbedding(x):
        return embed(tf.squeeze(tf.cast(x, tf.string)), signature='default', as_dict=True)['default']
    return UniversalEmbedding

# Squid model has a separate brain in each tentacle

def new_squid_model(embed, embed_size, num_symbols, num_syllables, optimizer='adam', dropout=0.5):
    input_text = layers.Input(shape=(1,), dtype=tf.string)
    embedding = layers.Lambda(define_embedding_function(embed), output_shape=(embed_size,), name='USE')(input_text)
    input_embeddings = layers.Input(shape=(embed_size,), dtype=tf.float32, name='Input')
    dense_input = layers.Dropout(dropout)(input_embeddings)
    dense = layers.Dense(1024, activation='relu', name='Convoluted')(dense_input)
    dense = layers.Dropout(dropout)(dense)
    dense = layers.Dense(2048, activation='relu', name='Midway')(dense)
    squid_names_array = []
    pred_array = []
    loss_array = []
    names_array = []
    for i in range(num_symbols):
      squid = layers.Dropout(dropout)(dense)
      squid = layers.Dense(256, activation='relu', name='Squid'+str(i))(squid)
      squid = layers.Dropout(dropout)(squid)
      name = 'Onehot'+str(i)
      pred_array.append(layers.Dense(num_syllables, activation='softmax', name=name)(squid))
      loss_array.append('categorical_crossentropy')
      names_array.append(name)
    model = Model(inputs=input_embeddings, outputs=pred_array)
    model.compile(loss=loss_array, 
                  optimizer=optimizer, 
                  metrics=['categorical_accuracy'])
    return model

# Add squid nature to transfer model
def add_squid_layers(model, num_symbols=0, num_syllables=0, optimizer='adam', dropout=0.5):
    pass

# Single output layer, just soaks up all syllables in multi-label configuration
# Used for training "transferable knowledge" about outputting to syllables.
def new_transfer_model(embed, embed_size=512, num_symbols=10, num_syllables=0, optimizer='adam', dropout=0.5):
    custom_binary_crossentropy = create_weighted_binary_crossentropy(1/(np.sqrt(num_syllables)), 1 - 1/np.sqrt(num_syllables))
    input_text = layers.Input(shape=(1,), dtype=tf.string)
    embedding = layers.Lambda(define_embedding_function(embed), output_shape=(embed_size,), name='USE')(input_text)
    dense_input = layers.Dropout(dropout)(embedding)
    dense = layers.Dense(1536, activation='relu', name='Human')(dense_input)
    dense = layers.Dropout(dropout)(dense)
    dense = layers.Dense(3072, activation='relu', name='Chimp')(dense)
    dense = layers.Dropout(dropout)(dense)
    #dense = layers.Dense(6144, activation='relu', name='Lizard')(dense)
    #dense = layers.Dropout(dropout)(dense)
    pred = layers.Dense(num_syllables, activation='sigmoid', name='Onehot')(dense)
    model = Model(inputs=[input_text], outputs=pred)
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['binary_crossentropy'])
    return model

# new_transfer_model, pop, new_test_model for new structure
def new_transfer_test(model, num_syllables=0, optimizer='adam', dropout=0.5):
    old_in = model.input
    old_out = model.layers[-1].output
    dense = layers.Dropout(dropout)(old_out)
    dense = layers.Dense(8192, activation='relu', name='Tuatara')(dense)
    dense = layers.Dropout(dropout)(dense)
    dense = layers.Dense(num_syllables, activation='sigmoid', name='Test')(dense)
    model2 = Model(old_in, dense)
    model2.compile(loss='binary_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['binary_crossentropy'])
    return model2

def save_squid_model(model):
    model.save_weights('./model_squid.h5')

def load_squid_model(model):
    model.load_weights('./model_squid.h5')  

def remove_squid_model():
    os.remove('./model_squid.h5')

# Joel Chao
def pop_layer(model):
    if not model.outputs:
        raise Exception('Model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

# above model has dropout, squidbrain, dropout, onehot for each symbol
def pop_squid_layers(model, num_symbols):
    for i in range(4 * num_sumbols):
        pop_layer(model)

def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def save_transfer_model(model):
    model.save_weights('./model_transfer.h5')

def load_transfer_model(model):
    model.load_weights('./model_transfer.h5')  

def freeze_transfer_model(model):
    for layer in model.layers:
        layer.trainable = False

def create_model_checkpoint(dir, model_name):
    filepath = dir + '/' + model_name + ".h5" 
    directory = os.path.dirname(filepath)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_weights_only=True, save_best_only=False)
    return checkpointer

def find_best_model(dir, model_name):
    progress = 0
    latest = None
    for file in os.listdir(dir):
        if file.startswith(model_name + '-'):
            epo = int(file.split('-')[1].split('.')[0])
            if epo > progress:
                progress = epo
                latest = file
    return (latest, progress)

def copy_model(source, target):
    with open(source, "rb") as inf:
        with open(target, "wb") as outf:
            while True:
                chunk = inf.read(8192)
                if not chunk:
                    break
                outf.write(chunk)
            outf.close()
            inf.close()

def create_model_copy_function(source_dir, target_dir, model_name):
    def epoch_end(epoch, logs):
        (latest_source, _) = find_best_model(source_dir, model_name)
        (latest_target, _) = find_best_model(target_dir, model_name)
        print("Callback: latest source, target = '{}', '{}'".format(latest_source, latest_target))
        if latest_source > latest_target:
            print("    copying")
            copy_model(source_dir + '/' + latest_source, target_dir + '/' + latest_target)
    return epoch_end

        #LambdaCallback(on_epoch_end=backup_best_model)

