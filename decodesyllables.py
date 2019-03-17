

from collections import Counter
import itertools

import cmudict
import arpabets
import pygtrie as trie

# mean score of number of phonemes before stopped
def sum_scores(arpa_list, score_list):
  sum = 0
  count = 0
  for i in range(len(arpa_list)):
    sum += score_list[i]
    count += 1
    if arpa_list[i] == '.':
      break
  return sum / count


class Decoder:
  
  # words for phoneme strings up to 10 long
  # dicts = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
  dicts = [{}] * 30
  multi_found = []

  # build reverse dict of phoneme->word, but only for words in our training set
  def __init__(self, cmudict):
    (self.singlemap, self.multimap) = cmudict.get_revmaps()
    self.reverse_dict = cmudict.get_reverse_dict()
    self.wordlist = set([])
    for word in open("blobs/wordlist", "r"):
      word = word[0:-1]
      self.wordlist.add(word)
    #print("wordlist: {0}".format(list(self.wordlist)[0:5]))
    #for key in self.singlemap.keys():
    #    print(key)
    #print(cmudict.singlemap['DH AH'])
    #print(cmudict.singlemap['DH AH'])
    self.reverse_dict = {}
    for key in self.singlemap:
      self.reverse_dict[key] = self.singlemap[key]
    for key in self.multimap:
      self.reverse_dict[key] = self.multimap[key]

  def decode_sentence(self, syll_list):
    poss_array = []

    # greedily walk forward, adding sub-trees of possible words
    def recurse(syll_list, curr_list, offset):
      end = len(syll_list)
      #print("checking {0}".format(syll_list[offset: offset + end+1]))
      if offset == len(syll_list):
        curr_list.append('.')
      found = False
      for i in range(0,end+1):
        sl = syll_list[offset:offset + i]
        if len(sl) == 0:
          continue
        #print("  {0} -> {1} = {2}".format(offset, offset + i, sl))
        key = "-".join(sl)
        if key in self.reverse_dict:
          found = True
          #print(" y: " + key)
          this_word = self.reverse_dict[key]
          next_list = [this_word, []]
          curr_list.append(next_list)
          recurse(syll_list, next_list[1], offset + i)
          if len(sl) > 1:
            word = self.reverse_dict[key]
            if not word in self.multi_found:
              print("    multi-syllable: " + word)
              self.multi_found.append(word)
        else:
          #print(" n: " + key)
          pass
      if not found:
        curr_list.append('!')

    # unwrap actual sentences into output array
    # recurse downward with current sentence, yield complete sentence if it ends with a period
    # [['the(2)', [['suh', ['!']], ['sun', [['litt', ['.']]]], ['sunlit', ['.']]]], ['thus', [['uhh', ['!']], ['un', [['litt', ['.']]]]]]]

    def walktree(lol, sentence):
      #print("Walk: {0}, sentence {1}".format(lol, sentence))
      for l in lol:
        #print("  check {0}".format(l))
        if type(l) == list:
           if type(l[0]) == type('text'):
             next = list(sentence)
             next.append(l[0])
             #print("    list, {0}".format(next))
             if len(l) > 1:
               for m in l[1:]:
                 for n in walktree(m, list(next)):
                   yield n
             else:
               yield next
           else:
             #print("    text, {0}".format(l))
             if l == '.':
               yield sentence
             elif l == '!':
               pass
             else:
               next = list(sentence)
               next.append(l)
               yield next
        else:
          #print("    text2, {0}".format(l))
          if l == '.':
            yield sentence
          elif l == '!':
            pass
          else:
            next = list(sentence)
            next.append(l)
            yield next

    recurse(syll_list, poss_array, 0)
    #print(poss_array)
    sentences = []
    min_len = len(syll_list) + 1
    for sentence in walktree(poss_array, []):
      sentence = list(sentence)
      if len(sentence) < min_len:
        min_len = len(sentence)
      sentences.append(" ".join(sentence))
    out = []
    for s in set(sentences):
      if len(s.split(' ')) == min_len:
        out.append(s)
    return out

if __name__ == "__main__":
  decoder = Decoder(cmudict.CMUDict())
  for x in decoder.decode_sentence(['DH AH', 'S AH N', 'L IH T']):
    print(x)
  print(' ')
  for x in decoder.decode_sentence(['DH AH', 'S AH N', 'L IH T', 'AA', 'N IH NG', 'HH IY', 'V IH NG', 'OW', 'V ER', 'HH EH D']):
    print(x)
