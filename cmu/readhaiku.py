from keras.preprocessing import text


import numpy as np

# haiku 5/7 syllable dataset reader
# format: long text tab short text
# return (long text as (N,""), short text as (N, ""), short syllables as (N, num_syll, 1))
class Reader:
    def __init__(self, word2sylls, decoder, wordmap):
        self.word2sylls = word2sylls
        self.decoder = decoder
        self.wordmap = wordmap
        textwordset = set()
        haikuwordset = set()

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
                    use_input = False
                _lastidx = -1
                for word in text.text_to_word_sequence(_haiku):
                    if word in self.word2sylls:
                        self.haikuwordset.add(word)
                        for syll in self.word2sylls[word]:
                            _sylls.append(syll)
                        _thisidx = self.decoder.word2idx[word]
                        #print('word {}, idx {}'.format(word, self.decoder.word2idx[word]))
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
                if len(big_text) == max_data:
                    break

        print('{} -> {} : {}'.format(big_text[0], big_haiku[0], big_data[0]))
        big_text = np.array(big_text)
        big_haiku = np.array(big_haiku)
        big_data = np.array(big_data)
        big_data = np.expand_dims(big_data, -1)
        return (big_text, big_haiku, big_data)

if __name__ == "__main__":
    from syllables_cmu import syllables as word2sylls
    from mappers import Decoder
    from wordmap import Wordmap

    max_words = 1000000
    decoder = Decoder(word2sylls)
    wordmap = Wordmap(max_words)
    reader = Reader(word2sylls, decoder, wordmap)
    (big_text, big_haiku, big_data) = reader.readfile("haiku_5.txt")
    print('{} -> {} : {}'.format(big_text[0], big_haiku[0], big_data[0]))

    print('Full length clauses: ', len(big_text))

