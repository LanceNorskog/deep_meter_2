from keras.preprocessing import text


import numpy as np

# haiku 5/7 syllable dataset reader
# format: long text tab short text
# return (long indexes as (N,[word indexes] short text as (N, [word indexes]))
# build word index while reading. Later, map to indexes.
class Reader:
    def __init__(self, word2sylls, decoder):
        self.word2sylls = word2sylls
        self.decoder = decoder
        self.textwordset = set()

    def readfile(self, haikufile, max_words=10, max_data=10000000, duplicate_haiku=True, deduplicate_haiku=True):
        big_text = []
        big_haiku = []
        with open(haikufile) as f:
            last_haiku = ''
            unknown = set()
            for line in f.readlines():
                _parts = line.strip().split('\t')
                _text = _parts[0]
                _haiku = _parts[1]
                _use_input = True
                if deduplicate_haiku and _haiku == last_haiku:
                    continue
                textwords = list(text.text_to_word_sequence(_text))
                if len(textwords) > max_words:
                    _use_input = False
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
                fail = False
                for word in words:
                    if not word in self.word2sylls:
                        fail = True
                        if not word in unknown:
                            print("Haiku word not in word2sylls: " + word)
                            unknown.add(word)
                if fail:
                    continue
                _haiku = ' '.join(words)
                for textword in textwords:
                    self.textwordset.add(textword)
                if _use_input:
                    big_text.append(_text)
                    big_haiku.append(_haiku)
                if duplicate_haiku:
                    big_text.append(_haiku)
                    big_haiku.append(_haiku)
                last_haiku = _haiku
                if len(big_text) >= max_data:
                    break

        big_text = np.array(big_text)
        big_haiku = np.array(big_haiku)
        return (big_text, big_haiku)

    # use "hashing trick" for input indexes
    # could be long text or haiku
    def gethash(self, big_input, max_words=10, hash_mole=20000):
        hashed = np.ones((len(big_input), max_words + 2), dtype='float32')
        for i in range(len(big_input)):
            hashed[i][0] = 0
            j = 1
            for h in text.hashing_trick(big_input[i], hash_mole - 2, hash_function='md5'):
                if j == max_words:
                    print('haiku too long? ', big_input[i])
                hashed[i][j] = h
                j += 1
        return hashed

    def getindexes(self, big_input, max_words=10):
        indexes = np.ones((len(big_input), max_words + 2), dtype='int32')
        for i in range(len(big_input)):
            indexes[i][0] = 0
            j = 1
            for w in big_input[i].split():
                if j == max_words:
                    print('haiku too long? ', big_input[i])
                indx = self.decoder.word2idx[w]
                indexes[i][j] = indx
                j += 1
        return indexes
        

if __name__ == "__main__":
    from syllables_cmu import syllables as word2sylls
    from mappers import Decoder

    # max input length
    max_words = 40
    total_words = 1000000
    decoder = Decoder(word2sylls)
    reader = Reader(word2sylls, decoder)
    (big_text, big_haiku) = reader.readfile("haiku_5.txt", max_words=max_words)
    print('{} -> {}'.format(big_text[0], big_haiku[0]))
    print('... {}'.format(reader.gethash(big_text, max_words, 10000)[0]))
    print('... {}'.format(reader.getindexes(big_haiku, 5)[0]))

    print('Full length clauses: ', len(big_text))

