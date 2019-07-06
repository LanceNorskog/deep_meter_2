import numpy as np

# no language model

class Decoder:
    def __init__(self, word2sylls):
        self.word2sylls = word2sylls
        big_sylls = set()

        self.wordlist = [''] * (len(word2sylls))
        self.wordlength = [0] * (len(word2sylls))
        for index, word in enumerate(word2sylls):
            self.wordlist[index] = word
            sylls = word2sylls[word]
            self.wordlength[index] = len(sylls)
            for syll in sylls:
                big_sylls.add(syll)

        self.syll2idx = {}
        for index, syll in enumerate(big_sylls):
            self.syll2idx[syll] = index
        self.idx2syll = [0] * len(self.syll2idx)
        for index, syll in enumerate(self.syll2idx):
            self.idx2syll[index] = syll

        # indexes into wordlist
        # 5 is longest known word 
        # idx2word[0][1] = word index means "syll #2 is first syll in word [index]"
        self.idx2word = np.zeros((2, len(self.idx2syll)), dtype='int32')
        self.idx2word = self.idx2word - 1
        for index, word in enumerate(self.wordlist):
            j = 0
            for syll in word2sylls[word]:
                self.idx2word[j][self.syll2idx[syll]] = index
                j += 1

    ''' given set of n-syllable indexes, return sentences or nulls'''
    def get_sentences(self, predictions):   
        print('predictions: ', predictions[0])
        out = [[]] * len(predictions)
        num_sylls = len(predictions[0])
        max_sylls = 2
        for i in range(len(predictions)):
            words = []
            predict = predictions[i]
            print('pred: ', predict)
            j = 0
            while j < num_sylls:
                word_ind = self.idx2word[0][predict[j]]
                print('word[{}]: {}'.format(j, word_ind))
                if word_ind >= 0:
                    words.append(self.wordlist[word_ind])
                    k = 1
                    while k < max_sylls and j+k < num_sylls:
                        print('k: {}, pred: {}'.format( k, predict[j+k]))
                        if word_ind == self.idx2word[k][predict[j+k]]:
                            print('skip! ')
                            j += 1
                        else:
                            break
                        k += 1
                else:
                    print('fail')
                    words = []
                    break
                j += 1
            out[i] = words
        return out
                
        



if __name__ == "__main__":
    syllables = {'the':['DH AH'], 'mugger':['M AH', 'G ER'], 'is': ['IH Z'], 'here':['HH IH R']}
    decoder = Decoder(syllables)
    print(decoder.syll2idx.keys())
    print(decoder.syll2idx.values())
    print(decoder.syll2idx['DH AH'], decoder.idx2syll[0])
    print('# features: ', len(decoder.idx2syll))

    print('wordlist: ', decoder.wordlist)
    print('idx2word: ', decoder.idx2word)
    
    predict = [[decoder.syll2idx['DH AH'], decoder.syll2idx['M AH'], decoder.syll2idx['G ER'], decoder.syll2idx['IH Z'], decoder.syll2idx['HH IH R']]]
    print(decoder.get_sentences(predict))
    predict = [decoder.syll2idx['M AH']]]
    print(decoder.get_sentences(predict))
    predict = [decoder.syll2idx['G ER']]]
    print(decoder.get_sentences(predict))
    predict = [-1]]
    print(decoder.get_sentences(predict))

    # decode 5 ints into a series of words
    # idx2word[0-4][index into wordlist]
    
