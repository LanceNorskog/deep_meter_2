
# generate tokenizer that understands small set of 5-syllable haiku lines only
# for use as output, not input

from keras.preprocessing import text
import json

from cmu.syllables_cmu import syllables as word2sylls

def tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.
    # Arguments
        json_string: JSON string encoding a tokenizer configuration.
    # Returns
        A Keras Tokenizer instance
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = text.Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer

max_features=100000
toki_file = './toki_cmu.json'

def haiku_tokenizer():
    with open(toki_file, "r") as f:
        return tokenizer_from_json(f.read())


# build the tokenizer for haiku
if __name__ == '__main__':
    toki = text.Tokenizer()
    haiku_data = 'haiku_5_same.txt'
    with open(haiku_data) as f:
        for line in f.readlines():
            words = text.text_to_word_sequence(line.split('\t')[0])
            for w in words:
                for w2 in w.split("'"):
                    if w2 == '':
                        continue
                    
                if w2 in word2sylls:
                    toki.fit_on_texts(word2sylls[w2])
                else:
                    print('not a word: ', w2)
            
    with open(toki_file, "w") as f:
        f.write(toki.to_json())

    toki2 = haiku_tokenizer()

