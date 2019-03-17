# TODO - change cmudict.get_syllables to get variant syllables that have same meter, add to emitted lines as separate lines so that randomizing works
# TODO - write randomizer, train/dev/test split
# TODO - emits 9-syllable lines, handle comma at end and beginning
# TODO - two-syllable trochees can be one syllable, figure out rules
#         DU-du du becomes (DU-du) du
# TODO - allow 9-syllable lines, add pause at beginning or end to match

from __future__ import absolute_import, division, unicode_literals

import sys
import cmudict
import meter
import tokens
import string
import collections

prefix = "gutenberg."
#meters = {"iambic_pentameter": "0101010101", "hiawatha": "10101010"}
meters = {"iambic_pentameter": "0101010101"}

cmudict = cmudict.CMUDict()
meter = meter.meter(cmudict, meters, "word_not_found")

def deb(x):
  #print(str(x) + "\n")
  pass

outputs = {}
for name in meters.keys():
  outputs[name] = open(prefix + name, "w")

failed_list = open("failed_meter.txt", "w")
failed_list = open("/dev/null", "w")
def fail(line, words):
  failed_list.write(line + "\t" + str(words) + "\n")

total = 0
# found one or more meters
correct = 0
# did not find a meter
failed = 0
# total number of meters guessed
guessed = 0

def filter(line):
  amp = False
  for i in range(len(line)):
    if line[i] == '&':
      amp = True
    elif line[i] == ';' and amp:
      return False
  return True

# one possible set of meters for this line
# words are (2) form
def do_possible(line, words, poss, saved):
  global guessed
  global failed
  stressarray = meter.getstress(poss)
  guesses = meter.meter_strict(stressarray)
  if len(guesses) == 0:
    failed += 1
    return 
  deb(line + "->" + str(guesses))
  for guess in guesses:
    if not guess in saved:
      sylls = []
      for word in words:
         # [ [ 'syllable', 'alternate'], ...]
         word_sylls = cmudict.get_syllables(word)
         sylls.append(word_sylls)
      outputs[guess].write(line + "\t" + str(sylls) + "\n")
      saved.append(guess)
      guessed += 1


# all possible meter sets for this line
def do_possibles(line, words, possibles):
  global correct
  global failed
  # incrementally remove pauses if no luck
  if len(possibles) == 0 and words[:-1] == ",":
    words = words[0:-1]
    possibles = meter.possibles(words, cmudict.syll_dict)
  while len(possibles) == 0 and "," in words:
    words = list(words)
    words.remove(",")
    possibles = meter.possibles(words, cmudict.syll_dict)
  if len(possibles) == 0:
    fail(line, str(words))
    failed += 1
    return
  # only save a line once per guessed meter
  failed_poss = 0
  saved = []
  for poss in possibles:
    do_possible(line, words, poss, saved)
  if len(saved) > 0:
    correct += 1
  else:
    failed += 1
    fail(line, str(possibles[0]))

for line in sys.stdin:
  if not filter(line):
    continue
  total += 1
  line = tokens.clean(line)
  words = tokens.tokenize(line)
  words = tokens.fixtokens(words)
  words = tokens.hyphen(words, cmudict.syll_dict)
  possibles = meter.possibles(words, cmudict.syll_dict)
  deb(line + " -> " + str(words) + " -> " + str(possibles))
  do_possibles(line, words, possibles)

for (name, f) in outputs.items():
  f.close()

sys.stderr.write("Total (filtered): {0}, correct: {1}, guessed {2}, failed: {3}\n".format(total, correct, guessed, failed))
sys.stderr.flush()
