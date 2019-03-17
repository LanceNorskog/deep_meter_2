# load digested corpus, build language model object, and pickle

import sys
import pickle
from ast import literal_eval

import syllables
import languagemodel
import cmudict
import tokens

# build languagemodel of word -> [enc, enc, ...]

sm = languagemodel.SyllableModel()
cmudict = cmudict.CMUDict()
syll_mgr = syllables.syllables()

print('Starting')
count = 0
for line in sys.stdin:
    parts = line.split('\t')
    text = parts[0]
    syllables = literal_eval(parts[1][:-1])  # newline
    text = tokens.clean(text)
    words = tokens.tokenize(text)
    words = tokens.fixtokens(words)
    words = tokens.hyphen(words, cmudict.syll_dict)
    clean = []
    for word in words:
        if word != ',':
            clean.append(word)
    count += 1
    if len(clean) != len(syllables):
        print("Line #: " + str(count))
        print(clean)
        print(syllables)
        continue
    for i in range(len(clean)):
        encs = []
        for s in syllables[i]:
            encs.append(syll_mgr.get_encoding(s))
        sm.addWord(clean[i], encs)

print('Saving')
sm.saveModel()
    
