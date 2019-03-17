
from sets import Set
import sys

import cmudict
import tokens

cmudict = cmudict.CMUDict()

wordset = Set([])
for line in sys.stdin:
  text = line.split("\t")[0]
  words = tokens.tokenize(line)
  words = tokens.fixtokens(words)
  words = tokens.hyphen(words, cmudict.syll_dict)
  for word in words:
    wordset.add(word)

for word in sorted(wordset):
  print(word)
