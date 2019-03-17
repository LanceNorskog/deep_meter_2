# stolen from https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
# with language model stolen from Harald Sheidl:
# https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7

from math import log
import numpy as np

# beam search: [[0.1, 0.2, ... # syllables], ... # words], beam width, optional language model
# figure out how to look ahead and score pairs!
# Maybe do individual score first, then walk again?
def word_beam_search_decoder(data, k, lm=None):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            if lm is not None:
                for j in range(len(row)-1):
                    lm_score = lm.getBigramProb(j
                    candidate = [seq + [j], score * -log(row[j])]
                    all_candidates.append(candidate)
            else:
                for j in range(len(row)):
                    candidate = [seq + [j], score * -log(row[j])]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences

if __name__ == "__main__":
    # define a sequence of 10 words over a vocab of 5 words
    data = np.random.random((10,5))
    data = np.array(data)
    # decode sequence
    result = beam_search_decoder(data, 7)
    # print result
    for seq in result:
        print(seq)
    sorted = np.sort(result, axis=1)
    for seq in sorted:
        print(seq)
