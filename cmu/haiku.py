
import numpy as np
from keras.preprocessing import text


def read_haiku(big_data_file, word2sylls, max_words=None, max_data=None, deduplicate_haiku=True):
    big_text = []
    big_haiku = []
    big_data = []
    textwordset = set()
    haikuwordset = set()
    with open(big_data_file) as f:
        last_haiku = ''
        for line in f.readlines():
            _parts = line.strip().split('\t')
            _text = _parts[0]
            _haiku = _parts[1]
            _sylls = []
            if deduplicate_haiku and _haiku == last_haiku:
                continue
            words = list(text.text_to_word_sequence(_haiku))
            if len(words) > max_words:
                continue
            for word in text.text_to_word_sequence(_haiku):
                if word in word2sylls:
                    haikuwordset.add(word)
                    for syll in word2sylls[word]:
                        _sylls.append(syll)
            for textword in list(text.text_to_word_sequence(_text)):
                textwordset.add(textword)
            if len(_sylls) != 5:
                continue
            _data = np.zeros((5), dtype='int32')
            for j in range(5):
                 _data[j] = syll2idx[_sylls[j]]
            big_text.append(_text)
            big_haiku.append(_haiku)
            big_data.append(_data)

    print("big_data: ", big_data)
            
    big_text = np.array(big_text)
    big_haiku = np.array(big_haiku)
    big_data = np.array(big_data)
    big_data = np.expand_dims(big_data, -1)

    shuffle = np.arange(len(big_text))
    print(shuffle)
    np.random.shuffle(shuffle)
    shuffle = shuffle[0:max_data]
    big_text = big_text[shuffle]
    big_haiku = big_haiku[shuffle]
    big_data = big_data[shuffle]

    (big_haiku, big_text, big_data)

if __name__ == "__main__":
    (h, t, d) = read_haiku("haiku_5.txt", {"haiku":['the']}, max_words=8, max_data=1000, deduplicate_haiku=True)
    print('# records: ', len(h))
