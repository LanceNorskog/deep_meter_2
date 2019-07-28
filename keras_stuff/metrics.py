import keras.backend as K
import keras.metrics as metrics

def sparse_categorical_accuracy(y_true, y_pred):
    # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)
    # convert dense predictions to labels
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_pred_labels = K.cast(y_pred_labels, K.floatx())
    return K.cast(K.equal(y_true, y_pred_labels), K.floatx())

def sparse_categorical_accuracy_per_sequence(y_true, y_pred):
    # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)
    # convert dense predictions to labels
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_pred_labels = K.cast(y_pred_labels, K.floatx())
    return K.min(K.cast(K.equal(y_true, y_pred_labels), K.floatx()), axis=-1)

def sparse_temporal_top_k_categorical_accuracy(y_true, y_pred, k=5):
    original_shape = K.shape(y_true)
    y_true = K.reshape(y_true, (-1, K.shape(y_true)[-1]))
    y_pred = K.reshape(y_pred, (-1, K.shape(y_pred)[-1]))
    top_k = K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k)
    return K.reshape(top_k, original_shape[:-1])

def sparse_temporal_top_k_categorical_accuracy_per_sequence(y_true, y_pred, k=5):
    original_shape = K.shape(y_true)
    y_true = K.reshape(y_true, (-1, K.shape(y_true)[-1]))
    y_pred = K.reshape(y_pred, (-1, K.shape(y_pred)[-1]))
    top_k = K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k)
    perfect = K.min(K.cast(top_k, 'int32'), axis=-1)
    return perfect #K.expand_dims(perfect, axis=-1)

def sparse(y_true, y_pred):
    return sparse_categorical_accuracy(y_true, y_pred)
def sparse1(y_true, y_pred):
    return sparse_temporal_top_k_categorical_accuracy(y_true, y_pred, k=1)
def perfect(y_true, y_pred):
    return sparse_categorical_accuracy_per_sequence(y_true, y_pred)
def perfect1(y_true, y_pred):
    return sparse_temporal_top_k_categorical_accuracy_per_sequence(y_true, y_pred, k=1)
def sparse5(y_true, y_pred):
    return sparse_temporal_top_k_categorical_accuracy(y_true, y_pred, k=5)
def perfect5(y_true, y_pred):
    return sparse_temporal_top_k_categorical_accuracy_per_sequence(y_true, y_pred, k=5)
def fscore(y_true, y_pred):
    recall = K.mean(sparse_categorical_accuracy(y_true, y_pred))
    precision = K.mean(sparse_categorical_accuracy_per_sequence(y_true, y_pred))
    return 2 * ((recall * precision)/(recall + precision))

def sparse_loss(y_true, y_pred):
    return scc(y_true, y_pred)

def perfect_loss(y_true, y_pred):
    return scct(y_true, y_pred, scale=1.0)
