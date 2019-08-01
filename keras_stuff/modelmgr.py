
import tensorflow as tf
from keras import layers 
from keras import callbacks 
from keras.models import Model

from keras_multi_head import MultiHeadAttention
from keras_pos_embd import PositionEmbedding

# handle base model, load variations from base via names and freeze the base
class ModelManager:
    embedding_name='embedding_1'
    bidirectional_name='bidirectional_1'
    cu_dnnlstm_name='cu_dnnlstm_2'
    dense_name='dense_1'

    def __init__(self):
        self.base_names = [
            self.embedding_name,
            self.bidirectional_name,
            self.cu_dnnlstm_name,
            self.dense_name,
        ]
        
        pass

    # if given names, load only those names and optionally freeze them
    def load_weights(self, model, filename, freeze=False):
        if freeze:
            model.load_weights(filename, by_name=True)
            for layer in model.layers:
                if layer.name in self.base_names:
                    layer.trainable=False
                    print('Freezing layer: ', layer.name)
                else:
                    print('Layer name not found: ', layer.name)
        else:
            model.load_weights(self.filename)

    def get_lstm(self, size, return_sequences=True, name='monkeys'):
        if tf.test.is_gpu_available():
            return layers.CuDNNLSTM(size, return_sequences=return_sequences, name=name)
        else:
            return layers.LSTM(size, return_sequences=return_sequences, name=name)

    def unfreeze(self, model, name):
            for layer in model.layers:
                if layer.name == name:
                    layer.trainable=True
                    print('Unfreezing layer: ', layer.name)

    # generate variations of model for experimentation
    def get_model(self, params, a=False, b=False, c=False, d=False, e=False, f=False, g=False, dropout=0.5):
        hash_input = layers.Input(shape=(params['max_words'],), dtype='int32')
        x = layers.Embedding(params['hash_mole'], params['embed_size'], input_length=params['max_words'], name=self.embedding_name)(hash_input)
        x = layers.Dropout(dropout/3)(x)
        if a:
            # did not train
            # needs positional embedding?
            x = MultiHeadAttention(4)(x)
            x = layers.Dropout(dropout/3)(x)
        if b:
            x = layers.Bidirectional(self.get_lstm(params['units']//2, return_sequences=True))(x)
            x = layers.Dropout(dropout)(x)
            x = layers.TimeDistributed(MultiHeadAttention(4))(x)
            #x = layers.Flatten()(x)
            #x = layers.Dropout(dropout)(x)
            #x = layers.Dense(params['embed_size'])(x)
            x = layers.Dropout(dropout/3)(x)
        #if c:
        x = layers.Bidirectional(self.get_lstm(params['units']//2, return_sequences=False, name=self.bidirectional_name))(x)
        x = layers.Dropout(dropout)(x)
        x = layers.RepeatVector(params['num_sylls'])(x)
        x = layers.Dropout(dropout)(x)
        if d:
            x = PositionEmbedding(
                input_dim=params['embed_size'],
                output_dim=params['num_sylls']*4,
                mode=PositionEmbedding.MODE_CONCAT
            )(x)
            x = layers.Dropout(dropout)(x)
            x = MultiHeadAttention(4)(x)
            x = layers.Dropout(dropout)(x)
        x = self.get_lstm(params['units'], return_sequences=True, name=self.cu_dnnlstm_name)(x)
        if e:
            x = PositionEmbedding(
                input_dim=params['units'],
                output_dim=params['units'],
                mode=PositionEmbedding.MODE_ADD
            )(x)
            x = layers.Dropout(dropout)(x)
            #x = layers.Dense(embed_size)(x)
            #x = layers.Dropout(dropout)(x)
        if f:
            # this was somewhat effective
            x = MultiHeadAttention(2)(x)
            x = layers.Dropout(dropout)(x)
        if g:
            x = layers.Dense(embed_size, kernel_initializer='identity')(x)
            x = layers.Dropout(dropout)(x)
        output_layer = layers.Dense(params['max_features'], activation='softmax', name=self.dense_name)(x)
        model = Model(inputs=[hash_input], outputs=[output_layer])
        return model

# for driving learning
class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#if False:
#    model = Sequential()
#    model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
#    model.add(Activation('softmax'))
#    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#
#    history = LossHistory()
#    model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])
#
#    print(history.losses)


if __name__ == "__main__":
    names = [
        'embedding_1',
        'bidirectional_1',
        'cu_dnnlstm_2',
        'dense_1',
        ]
    params = {
        'max_words':10 , 'hash_mole': 20000, 'embed_size': 512, 'units': 512, 'num_sylls': 5, 'max_features': 17000
        }
    model_file = "/tmp/haiku_zhg_mha_5.h5"
    #modelmgr = ModelManager()
    modelmgr = ModelManager()
    model = modelmgr.get_model(params, f=True)
    modelmgr.load_weights(model, model_file, freeze=True)
    model.summary()
