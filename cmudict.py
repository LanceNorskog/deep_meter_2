import collections
import pygtrie as trie
import ast

# load cmudict-syll_dict
# dict {word -> [syllable list]

# from NLTK, overkill
# need arpabets for words not in cmulist!

#stopword_1 = ["i", "me", "my", "we", "our", "ours", "you", "you're", "you've", "you'll", "you'd", "your", "yours", "he", "him", "his", "she", "she's", "her", "hers", "it", "it's", "its", "they", "them", "their", "theirs", "what", "which", "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "a", "an", "the", "and", "but", "if", "or", "as", "of", "at", "by", "for", "with", "through", "to", "from", "up", "down", "in", "out", "on", "off", "then", "once", "here", "there", "when", "where", "why", "how", "all", "both", "each", "few", "more", "most", "some", "such", "no", "nor", "not", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", "should", "now", "d", "ll", "m", "o", "re", "aren't", "shan", "shan't", "wasn", "won", "won't", ","]

cmudict = "/Users/l0n008k/open/data/cmudict_0.6.syllablized.txt"
cmudict = "/home/lance/open/data/cmudict_0.6.syllablized.txt"

# hell maybe this should be a trie also
def load_dictionary():
  syll_dict = {}
  stress_dict = {}
  reverse_dict = {}
  #for stopword in stopword_1:
  #  stress_dict[stopword] = [ "1" ]
  with open(cmudict, "r") as ins:
    for line in ins:
      if line.startswith("#"):
        continue
      words = line.split(" ")
      key = words[0].lower()
      rest = words[2:]
      ph_list = []
      dbg = False
      if line.startswith("THE "):
        dbg = True
      for ph in rest:
        if ph != '-':
          if ph.endswith("\n"):
            ph = ph[:-1]
          if ph.endswith("0") or ph.endswith("1") or ph.endswith("2"):
            ph = ph[:-1]
            if dbg:
              print("ph = " + ph)
          ph_list.append(ph)
      phonemes = " ".join(ph_list)
      reverse_dict[phonemes] = key
      syllarray = []
      stressarray = []
      syll = ""
      last = ""
      for arpa in rest:
        if arpa.endswith("\n"):
          arpa = arpa[:-1]
        if arpa.endswith("0") or arpa.endswith("2"):
          stressarray.append("0")
          arpa = arpa[:-1]
        elif arpa.endswith("1"):
          stressarray.append("1")
          arpa = arpa[:-1]
        if arpa == "-" and len(syll) > 0:
          syllarray.append(syll)
          syll = ""
        elif last == "-":
          syll = arpa
        elif syll == "":
          syll = arpa
        else:
          syll = syll + " " + arpa
        last = arpa
      syllarray.append(syll)
      syll_dict[key] = syllarray
      stress_dict[key] = stressarray
  return (syll_dict, stress_dict, reverse_dict)

# Trie of syllable sequences to word
# All syllable sequences of word point yield the base word
def loadRevmap(syll_dict):
  #print("making reverse maps")
  singlemap = trie.StringTrie(separator='-')
  multimap = trie.StringTrie(separator='-')
  i = 0
  for key in syll_dict.keys():
    syllarray = syll_dict[key]
    if len(syllarray) > 1:
      # 'M AH-G ER' -> "mugger"
      multimap['-'.join(syllarray)] = key
    else:
      singlemap[syllarray[0]] = key
    i += 1
  return (singlemap, multimap)

def load_blobs():
  f_syll = open("blobs/syllables.dict", "r")
  line = f_syll.read()
  f_syll.close()
  syll_dict = ast.literal_eval(line)
  f_stress = open("blobs/stresses.dict", "r")
  line = f_stress.read()
  stress_dict = ast.literal_eval(line)
  f_stress.close()
  return (syll_dict, stress_dict)

def save_blobs(syll_dict, stress_dict):
  f_syll = open("blobs/syllables.dict", "w")
  f_syll.write(str(syll_dict))
  f_syll.close()
  f_stress = open("blobs/stresses.dict", "w")
  f_stress.write(str(stress_dict))
  f_stress.close()

class CMUDict():

  def __init__(self):
    self.syll_dict = {}
    self.stress_dict = {}
    self.reverse_dict = {}
    (self.syll_dict, self.stress_dict) = load_blobs()
    # faster to make this than read it!
    self.singlemap = {}
    self.multimap = {}
    (self.singlemap, self.multimap) = loadRevmap(self.syll_dict)

  # get all syllable sets for given word
  # [ [ 'M AH', 'G ER'], ['M U', 'GER'] ]
  def get_syllables(self, word):
    out = []
    for suffix in [ '', '(2)', '(3)', '(4)', '(5)', '(6)' ]:
      word_suf = word + suffix
      if word_suf in self.syll_dict:
        word_syll = self.syll_dict[word_suf]
        if not word_syll in out:
          out.append(word_syll)
          break
    return out

  def get_revmaps(self):
    return (self.singlemap, self.multimap)
  
  def get_reverse_dict(self):
    return self.reverse_dict
  
#(syll_dict, stress_dict) = load_dictionary()
#print(syll_dict["mugger"])
#print(stress_dict["mugger"])

#revmap = loadRevmap(syll_dict)
#print(revmap)
#print(revmap['M AH-G ER'])
#x = collections.Counter()

if __name__ == "__main__":
  print("Reading source dictionary")
  (syll_dict, stress_dict, reverse_dict) = load_dictionary()
  print(reverse_dict['DH AH'])
  print("Creating blobs literals")
  save_blobs(syll_dict, stress_dict)
  print("Loading blobs literals")
  cd = CMUDict()
  print(cd.syll_dict['the'])
  print(cd.syll_dict['mugger'])
  print(cd.get_syllables('the'))
  print(cd.get_syllables('mugger'))
  print(cd.stress_dict['the'])
  print(cd.stress_dict['mugger'])
  print(cd.singlemap['DH AH'])
  print(cd.multimap['M AH-G ER'])
