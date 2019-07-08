import numpy as np

# no language model

class Decoder:
    def __init__(self, word2sylls):
        self.word2sylls = word2sylls
        big_sylls = set()

        self.wordlist = [''] * (len(word2sylls) + 10)
        self.wordlength = [0] * (len(word2sylls) + 10)
        for index, word in enumerate(word2sylls):
            self.wordlist[index + 10] = word
            sylls = word2sylls[word]
            self.wordlength[index + 10] = len(sylls)
            for syll in sylls:
                big_sylls.add(syll)

        num_sylls = len(big_sylls)
        # longest word
        max_sylls = 5
        self.syll2idx = {}
        for index, syll in enumerate(big_sylls):
            self.syll2idx[syll] = index
        self.idx2syll = [0] * len(self.syll2idx)
        for index, syll in enumerate(self.syll2idx):
            self.idx2syll[index] = syll

        # indexes into wordlist
        # 5 is longest known word 
        # idx2word[0][1] = [word index] means "syll #2 is first syll in word [index]"
        self.idx2word = np.zeros((2, len(self.idx2syll)), dtype='int32')
        self.idx2word = [[]] * max_sylls
        for i in range(max_sylls):
            self.idx2word[i] = [[]] * num_sylls
            for j in range(num_sylls):
                self.idx2word[i][j] = []
        for index, word in enumerate(self.wordlist):
            if index < 10:
                continue
            j = 0
            for syll in word2sylls[word][:max_sylls]:
                self.idx2word[j][self.syll2idx[syll]].append(index)
                j += 1
        self.wordlist = np.array(self.wordlist)

    def get_partial_sentences(self, predict, partial, step):
        def noise(s):
            indent = ''.join(['.' for i in partial])
            #print(indent, s)

        noise('predict: {}, partial {}'.format(predict, partial))
        num_sylls = len(predict)
        if step == num_sylls:
            noise('yielding: {}'.format(partial))
            yield partial
            return
        max_sylls = 2
        wordset = self.idx2word[0][predict[step]]
        noise('wordset[{}]: {}'.format(step, wordset))
        for word_ind in wordset:
            noise('word[{}]: {}'.format(step, word_ind))
            k = 0
            while k < max_sylls and step+k < num_sylls:
                if not word_ind in self.idx2word[k][predict[step+k]]:
                    break
                noise('k: {}, pred: {}'.format(k, predict[step+k]))
                k += 1
            if k == self.wordlength[word_ind]:
                words = partial.copy()
                words.append(word_ind)
                for x in self.get_partial_sentences(predict, words, step + k):
                    noise('passing {}'.format(x))
                    yield x
                return

    ''' given set of n-syllable indexes, return word index lists or nulls'''
    def get_sentences(self, predictions):   
        out = [[]] * len(predictions)
        num_sylls = len(predictions[0])
        max_sylls = 2
        for i in range(len(predictions)):
            out[i] = []
            for x in self.get_partial_sentences(predictions[i], [], 0):
                out[i].append([x])
        return out

    def decode_sentences(self, sentences):
        out = []
        for words_list in sentences:
            sents = []
            for words in words_list:
                sents.append(self.wordlist[np.array(words)].tolist())
            out.append(sents)
        return out


if __name__ == "__main__":
    def test(decoder, haiku):
        print(haiku, ':')
        predict = []
        for arpa in haiku:
            predict.append(decoder.syll2idx[arpa])
        preds = decoder.get_sentences([predict])
        decs = decoder.decode_sentences(preds)
        for dec in decs:
            for poss in dec:
                for sent in poss:
                    print('  -> {}'.format(haiku, ' '.join(list(sent))))
        

    syllables = {'the':['DH AH'], 'mugger':['M AH', 'G ER'], 'is': ['IH Z'], 'here':['HH IH R']}
    decoder = Decoder(syllables)
    print(decoder.syll2idx.keys())
    print(decoder.syll2idx.values())
    print(decoder.syll2idx['IH Z'], decoder.idx2syll[0])
    print(decoder.wordlist)
    print('# features: ', len(decoder.idx2syll))

    print('wordlist: ', decoder.wordlist)
    print('idx2word: ', decoder.idx2word)

    test(decoder, ['IH Z'])
    test(decoder, ['IH Z', 'DH AH'])
    test(decoder, ['IH Z', 'DH AH', 'HH IH R'])
    test(decoder, ['DH AH', 'HH IH R', 'IH Z'])
    test(decoder, ['DH AH', 'M AH', 'G ER', 'IH Z', 'HH IH R'])
    test(decoder, ['M AH'])
    test(decoder, ['G ER'])
    
    predict = [[decoder.syll2idx['DH AH'], decoder.syll2idx['M AH'], decoder.syll2idx['G ER'], decoder.syll2idx['IH Z'], decoder.syll2idx['HH IH R']]]
    print('Predicted words: ', decoder.get_sentences(predict)[0])
    predict = [[decoder.syll2idx['M AH']]]
    print('Predicted words: ', decoder.get_sentences(predict))
    predict = [[decoder.syll2idx['G ER']]]
    print('Predicted words: ', decoder.get_sentences(predict))
    predict = [[decoder.syll2idx['G ER'], decoder.syll2idx['M AH']]]
    print('Predicted words: ', decoder.get_sentences(predict)[0])

    # decode 5 ints into a series of words
    # idx2word[0-4][index into wordlist]
    
