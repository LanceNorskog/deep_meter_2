import numpy as np

# no language model

class Decoder:
    def __init__(self, word2sylls):
        self.word2sylls = word2sylls
        big_sylls = set()

        self.wordlist = [''] * (len(word2sylls))
        for index, word in enumerate(word2sylls):
            self.wordlist[index] = word
            sylls = word2sylls[word]
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
        # idx2word[1][2] = word index means "syll #2 is first syll in word"
        self.idx2word = np.zeros((3, len(self.idx2syll)), dtype='int32')
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
        for i in range(len(predictions)):
            words = []
            for j in range(num_sylls):
                print('pred: ', predictions[i][j])
                word_ind = self.idx2word[predictions[i][j]][0]
                print(word_ind)
                if word_ind >= 0:
                    words.append(self.wordlist[word_ind])
                    skip = 0
                    for k in range(1, num_sylls - j):
                        print('k: {}, pred: {}'.format( k, predictions[i][j + k]))
                        if word_ind == self.idx2word[predictions[i][j + k]][k]:
                            print('skip! ')
                            skip += 1
                        else:
                            break
                    j += skip
                else:
                    pass  # should break out since can't start a word
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

    # decode 5 ints into a series of words
    # idx2word[0-4][index into wordlist]
    
