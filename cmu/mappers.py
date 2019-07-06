import numpy as np

# no language model

class Decoder:
    def __init__(self, word2sylls):
        self.word2sylls = word2sylls
        big_sylls = set()

        self.wordlist = [''] * len(word2sylls)
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
        self.idx2word = np.zeros((5, len(self.idx2syll)), dtype='int32')
        for index, word in enumerate(self.wordlist):
            j = 0
            for syll in word2sylls[word]:
                self.idx2word[j][self.syll2idx[syll]] = index


if __name__ == "__main__":
    syllables = {'the':['DH AH']}
    decoder = Decoder(syllables)
    print(decoder.syll2idx.keys())
    print(decoder.syll2idx['DH AH'], decoder.idx2syll[0])
    print('# features: ', len(decoder.idx2syll))


    # decode 5 ints into a series of words
    # idx2word[0-4][index into wordlist]
    
