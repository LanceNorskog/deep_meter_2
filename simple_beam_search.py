
# stolen from https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
# with language model stolen from Harald Sheidl:
# https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7

from math import log
import numpy as np
import sys
import languagemodel

# beam search
# [ [0.1, 0.1, 0.1, 0.1, 0.1] first word, [0.1, 0.1, 0.1, 0.1, 0.1] second word, ... []]
def beam_search_decoder(data, k, minProb=0.000000001):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                if row[j] >= minProb:
                    z = log(row[j])
                    if z < 0:
                        z = -z 
                    candidate = [seq + [j], score * z]
                    all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[-k:]
    return sequences

def Y_beam_search_decoder(data, k, minProb=0.000000001):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                if row[j] >= minProb:
                    candidate = [seq + [j], score * (0 - log(row[j]))]
                    all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[-k:]
    return sequences

def x_beam_search_decoder(data, k, minProb=0.000000001):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate

def x_beam_search_decoder(data, k, minProb=0.000000001):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
def beam_search_decoder(data, k, minProb=0.000000001):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                if row[j] >= minProb:
                    candidate = [seq + [j], score * (0 - log(row[j]))]
                    all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[-k:]
    return sequences

def x_beam_search_decoder(data, k, minProb=0.000000001):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                if row[j] >= minProb:
                    candidate = [seq + [j], score + (0 - log(row[j]))]
                    all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[-k:]
    return sequences

# beam search
def word_beam_search_decoder(data, k, lm):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            numWords = len(row)
            for enc in range(numWords):
                wscore = score
                wscore *= row[enc]
                scale = lm.getUnigramProb(row[0])
                next = lm.getNextSylls([enc])
                if len(next) > 0:
                    for word in next:
                        scale += lm.getBigramProb(row[enc-1], row[enc])
                else:
                    scale += lm.getUnigramProb(row[enc])
                if numWords > 1:
                    scale ** (1/(numWords+1))
                candidate = [seq + [enc], wscore * scale]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[-k:]
    return sequences

if __name__ == "__main__":
    # define a sequence of 3 words over a vocab of 5 words
    data=[
        [0.1, 0.2, 0.1, 0.1, 0.1],
        [0.3, 0.2, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.2, 0.7]]
    data = np.array(data)
    # decode sequence
    result = beam_search_decoder(data, 7)
    result2 = beam_search_decoder(data, 7)
    for i in range(len(result)):
      for j in range(len(result[0])):
        if result[i][0][j] != result2[i][0][j]:
          print("[{}][{}] -> ({}, {})".format(i, j, result[i][j], result2[i][j]))
          print('Fail!')
          sys.exit(1)
    print(result)
    sm = languagemodel.SyllableModel()
    sm.loadModel()
    result = word_beam_search_decoder(data, 2, sm)
    print(result)
