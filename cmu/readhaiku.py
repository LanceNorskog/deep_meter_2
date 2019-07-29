from keras.preprocessing import text


import numpy as np

# haiku 5/7 syllable dataset reader
# format: long text tab short text
# return (long indexes as (N,[word indexes] short text as (N, [word indexes]), short syllables as (N, num_syll, 1))
class Reader:
    def __init__(self, word2sylls, decoder, wordmap):
        self.word2sylls = word2sylls
        self.decoder = decoder
        self.wordmap = wordmap
        self.textwordset = set()
        self.haikuwordset = set()

    def readfile(self, haikufile, max_words=10, max_data=10000000, duplicate_haiku=True, deduplicate_haiku=True):
        big_text = []
        big_haiku = []
        big_data = []
        big_data_file = haikufile
        with open(big_data_file) as f:
            last_haiku = ''
            for line in f.readlines():
                _parts = line.strip().split('\t')
                _text = _parts[0]
                _haiku = _parts[1]
                _sylls = []
                _use_input = True
                if deduplicate_haiku and _haiku == last_haiku:
                    continue
                textwords = list(text.text_to_word_sequence(_text))
                if len(textwords) > max_words:
                    _use_input = False
                _lastidx = -1
                words = []
                for word in text.text_to_word_sequence(_haiku):
                    if word == "'":
                        continue
                    _word = None
                    if not word in self.word2sylls and word[-2:] == "'s":
                        if word[:-2] + 's' in self.word2sylls:
                            _word = word[:-2] + 's'
                        elif word[:-2] + 'es' in self.word2sylls:
                            _word = word[:-2] + 'es'
                        if _word:
                            word = _word
                    words.append(word)
                _haiku = ' '.join(words)
                for word in words:
                    if word in self.word2sylls:
                        self.haikuwordset.add(word)
                        for syll in self.word2sylls[word]:
                            _sylls.append(syll)
                        _thisidx = self.decoder.word2idx[word]
                        if _lastidx != -1:
                            self.wordmap.add(_lastidx, _thisidx)
                        _lastidx = _thisidx
                for textword in textwords:
                    self.textwordset.add(textword)
                if len(_sylls) != 5:
                    continue
                _data = np.zeros((5), dtype='int32')
                for j in range(5):
                     _data[j] = self.decoder.syll2idx[_sylls[j]]
                if _use_input:
                    big_text.append(_text)
                    big_haiku.append(_haiku)
                    big_data.append(_data)
                if duplicate_haiku:
                    big_text.append(_haiku)
                    big_haiku.append(_haiku)
                    big_data.append(_data)
                last_haiku = _haiku
                if len(big_text) >= max_data:
                    break

        big_text = np.array(big_text)
        big_haiku = np.array(big_haiku)
        big_data = np.array(big_data)
        # this kind of nonsense should be in keras model code
        big_data = np.expand_dims(big_data, -1)
        return (big_text, big_haiku, big_data)

    # use "hashing trick" for input indexes
    # could be long text or haiku
    def gethash(self, big_input, max_words=10, hash_mole=20000):
        hashed = np.zeros((len(big_input), max_words), dtype='float32')
        for i in range(len(big_input)):
            j = 0
            for h in text.hashing_trick(big_input[i], hash_mole, hash_function='md5'):
                if j == max_words:
                    print('haiku too long? ', big_input[i])
                hashed[i][j] = h
                j += 1
        return hashed

        

if __name__ == "__main__":
    from syllables_cmu import syllables as word2sylls
    from mappers import Decoder
    from wordmap import Wordmap

    # max input length
    max_words = 40
    total_words = 1000000
    decoder = Decoder(word2sylls)
    wordmap = Wordmap(total_words)
    reader = Reader(word2sylls, decoder, wordmap)
    (big_text, big_haiku, big_data) = reader.readfile("haiku_5.txt", max_words=max_words)
    print('{} -> {} : {}'.format(big_text[0], big_haiku[0], big_data[0]))
    print('... {}'.format(reader.gethash(big_text, max_words, 10000)[0]))
    print('... {}'.format(reader.gethash(big_haiku, 5, 10000)[0]))

    print('Full length clauses: ', len(big_text))

