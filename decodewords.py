
from collections import Counter
import itertools

import cmudict
import arpabets

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
  
  pho_list = ['IY', 'AW', 'DH', 'AY', 'HH', 'CH', 'JH', 'ZH', 'D', 'NG', 'TH', 'AA', 'B', 'AE', 'EH', 'G', 'F', 'AH', 'K', 'M', 'L', 'AO', 'N', 'IH', 'S', 'R', 'EY', 'T', 'W', 'V', 'Y', 'Z', 'ER', 'P', 'UW', 'SH', 'UH', 'OY', 'OW']
  
  # words for phoneme strings up to 10 long
  # dicts = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
  dicts = [{}] * 30

  # build reverse dict of phoneme->word, but only for words in our training set
  def __init__(self, reverse_dict, arpabets_mgr):
    self.reverse_dict = reverse_dict
    self.wordlist = set([])
    for word in open("blobs/wordlist", "r"):
      word = word[0:-1]
      self.wordlist.add(word)
    #print("wordlist: {0}".format(list(self.wordlist)[0:5]))
    single_rev = {}
    for key in reverse_dict:
      word = reverse_dict[key]
      if word in self.wordlist:
        single_rev[key] = word
    self.single_dict = single_rev
    #print(short_rev)
    phomap = {}
    for pho in self.pho_list:
      phomap[pho] = set([])
    #print("Phomap empty = " + str(phomap))
    for key in self.single_dict.keys():
      #print(key)
      good = True
      for pho in key.split(" "):
        if arpabets_mgr.get_encoding(pho) == 0:
          good = False
      if good:
        for pho in key.split(" "):
          word = self.reverse_dict[key]
          if word.endswith(")"):
            word = word[0:-3]
          if word in self.wordlist:
            phomap[pho].add(word)
        #print(key + " len = " + str(len(key)))
        phodict = self.dicts[key.count(" ")]
    for key in reverse_dict.keys():
      #print(key)
      good = True
      for pho in key.split(" "):
        if arpabets_mgr.get_encoding(pho) == 0:
          good = False
      if good:
        phodict = self.dicts[key.count(" ")]
        phodict[key] = reverse_dict[key]
    self.pho2words = phomap

  def decodewords(self, arpa_list):
    #print(arpa_list)
    first = 0
    last = 1
    words = Counter()
    while first < len(arpa_list):
      word = " ".join(arpa_list[first:last])
      #print(word)
      if word in self.reverse_dict:
        #print("{0} -> {1}".format(word, self.reverse_dict[word]))
        words[self.reverse_dict[word]] += 1
        first = last
        last = first + 1
      else:
        last = last + 1
    if first == len(arpa_list):
      return words
    else:
      return None

  def decode_sentence(self, arpa_list, max_word_len):
    poss_array = []

    def recurse(arpa_list, curr_list, word_num, offset, max_word_len):
      end = min(max_word_len, len(arpa_list) - offset)
      if end == 0:
        curr_list.append('.')
        return
      print("checking {0}".format(arpa_list[offset: offset + end+1]))
      found = False
      for i in range(0,end+1):
        sl = arpa_list[offset:offset + i]
        #print("  {0} -> {1} = {2}".format(offset, offset + i, sl))
        key = " ".join(sl)
        if key in self.dicts[i]:
          found = True
          #print(" y: " + key)
          this_word = self.dicts[i][key]
          next_list = [this_word, []]
          curr_list.append(next_list)
          recurse(arpa_list, next_list[1], word_num + 1, offset + i, max_word_len)
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

    recurse(arpa_list, poss_array, 0, 0, max_word_len)
    print(poss_array)
    sentences = []
    for sentence in walktree(poss_array, []):
      sentences.append(" ".join(sentence))
    out = []
    for s in set(sentences):
      out.append(s)
    return out

if __name__ == "__main__":
  print(sum_scores(['a', 'b', '.', 'd'], [0.6, 0.7, 0.8, 0.9]))
  def check1(phonemes):
    scores = [1] * len(phonemes)
    print(scores)
    wordlist = decoder.getwords('DH AH S AH N L IH T AA N IH NG HH IY V IH NG OW V ER HH EH D'.split(' '), scores)
    #print(len(wordlist))
  (x, y, reverse_dict) = cmudict.load_dictionary()
  decoder = Decoder(reverse_dict, arpabets.arpabets())
  for x in decoder.decode_sentence('DH AH S AH N L IH T AA N IH NG HH IY V IH NG OW V ER HH EH D'.split(' '), 20):
    print(x)
  
#'AE N D AO L OW L IH M P AH S R IH NG Z W IH DH L AW D AH L AA R M Z'
#'AE N HH AH M B AH L CH IH R F AH L HH AE P IY L AH V IH NG B AE N D'
#'P ER EY D IH NG IH N AH K AA M M AH JH EH S T IH K EH R'
#'DH AH K AA M ER S AH V DH AH W ER L D W IH DH T AA N IY L IH M'
#'DH AH W EY T AH V Y IH R Z AO R W ER L D L IY K EH R Z DH AE T P R EH S'
#'K AH N JH EH K CH ER AH V DH AH P L UW M AH JH AE N D DH AH F AO R M'
#'AE N D HH AE N D IH N HH AE N D DH AH L AE F IH NG B EH L AE D R EH S'
#'AH N T IH L DH AE T AW ER DH AH W AO R F EH R L AE S T AH D DH EH R'
#'AE N D R AE M B L IH NG B R AE M B AH L B EH R IY Z P AH L P AE N D S W IY T'
