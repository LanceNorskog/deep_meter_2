# Keras generator for text + sylls

# Read text lines, syllable-ize them.
# Return in batches to main loop
# Don't use 'yield', it does not multi-process well.

# Because of size concerns, tokenize each line of text on each epoch

# based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

# Only used for training Transfer brain with a multi-label output

from random import shuffle
import keras as K
import numpy as np
import os
import array
import pickle

import cmudict
import tokens
import cmudict
import syllables

# base test set
RAW_DIR='./full_raw'
CACHE_DIR='./full_data'
RAW_DIR='/content/full_raw'
CACHE_DIR='/content/full_data'
FILE_LINES=20000

total=0
success=0

# encode one line, if all syllables are in syllable dictionary
def encode_line(line, cmudict, syll_mgr):
    global total
    global success
    total += 1
    line = tokens.clean(line)
    words = tokens.tokenize(line)
    words = tokens.fixtokens(words)
    words = tokens.hyphen(words, cmudict.syll_dict)
    encs = array.array('H')
    for word in words:
         sylls = cmudict.get_syllables(word.lower())
         if sylls == None or len(sylls) == 0:
             continue
         for syll in sylls[0]:
             enc = syll_mgr.get_encoding(syll)
             if enc != syllables.unknown_encoding:
                 encs.append(enc)
    return encs


class DataGenerator(K.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, max_files=1000000, file_lines=FILE_LINES, batch_size=32, shuffle=True):
        'Initialization'
        self.raw_dir = RAW_DIR
        self.cache_dir = CACHE_DIR
        self.raw_files = []
        self.cache_files = []
        files = []
        for file in os.listdir(self.raw_dir):
            files.append(file)
            if len(files) >= max_files:
                break
        for file in files:
            self.raw_files.append(self.raw_dir + '/' + file)
        for file in files:
            self.cache_files.append(self.cache_dir + '/' + file + '.pk')
        # arange does this
        self.indexes = [1] * len(self.raw_files)
        for i in range(len(self.raw_files)):
            self.indexes[i] = i
        self.file_lines = file_lines
        self.batch_size = batch_size
        self.syll_mgr = syllables.syllables()
        self.n_classes = self.syll_mgr.get_size()
        self.shuffle = shuffle
        self.on_epoch_end()
        self.cmudict = cmudict.CMUDict()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.raw_files)

    def __getitem__(self, index):
        'Generate one batch of data'
        (text_array, encode_array) = self.read_cache(index)
        if text_array == None:
            (text_array, encode_array) = self.read_text(index)
            self.save_cache(index, text_array, encode_array)

        # Return in tensor-friendly format
        text_np = np.array(text_array)
        labels_np = np.zeros((len(encode_array), self.syll_mgr.get_size()), dtype=np.int8)
        for i in range(len(encode_array)):
            for enc in encode_array[i]:
                labels_np[i][enc] = 1
        if index == 0:
            print('index[{}]'.format(index))
        return text_np, labels_np

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.raw_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def read_text(self, index):
        with open(self.raw_files[self.indexes[index]], "r") as f:
            lines = f.read().splitlines()
        text_array = [] # one per accepted line
        encode_array = [] # one per multi-label onehot
        for line in lines:
            labels = encode_line(line, self.cmudict, self.syll_mgr)
            if labels != None:
                text_array.append(line)
                encode_array.append(labels)
        return (text_array, encode_array)

    # save and load format [['sentence',...],[encoding index,...]]
    def read_cache(self, index):
        if os.path.exists(self.cache_files[self.indexes[index]]):
            with open(self.cache_files[self.indexes[index]], "rb") as f:
                blob = pickle.load(f)
                return (blob[0], blob[1])
        else:
            return (None, None)

    def save_cache(self, index, text, labels):
        blob = [text, labels]
        with open(self.cache_files[self.indexes[index]], "wb") as f:
            pickle.dump(blob, f)

if __name__ == "__main__":
    gen = DataGenerator()
    print('Num batches: {}'.format(gen.__len__))
    (text, labels) = gen.__getitem__(0)
    print('Text: {}, {}'.format(text.shape, text[0]))
    print('Labels: {}, {}'.format(labels.shape, labels[0]))
    print('Total, success: {}, {}'.format(total, success))
