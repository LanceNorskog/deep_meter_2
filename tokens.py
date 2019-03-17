
from __future__ import absolute_import, division, unicode_literals
from itertools import product
from segtok.tokenizer import space_tokenizer, web_tokenizer
from hyphenate import hyphenate_word
import num2words
import string

# old, limited
def tokenize1(sentence):
  words = space_tokenizer(sentence)
  out = []
  for word in words:
    word = word.lower()
    if word.endswith(","):
      out.append(word[:-1])
      out.append(",")
    else:
      out.append(word)
  if out[-1] == ",":
    out = out[:-1]
  return out

# scrub some dirt out
def clean(sentence):
  sentence = sentence.replace("\"","")
  sentence = sentence.replace("\n","")
  sentence = sentence.replace("\\","")
  sentence = sentence.replace("'","")
  sentence = sentence.replace("`","")
  return sentence

def tokenize(sentence):
  words = web_tokenizer(sentence)
  #words = sentence.split(" ")
  out = []
  for word in words:
    word = word.lower()
    if word != "_":
      out.append(word)
  return out

#def tokenize(sentence):
#  words = sentence.split(" ")
#  return words

# dot, comma, semi become comma 
# numbers, cmudict only has spelled-out words
# maybe "pine-trees" should be a "10" stress, but create ["1", "1"]
def fixtokens(words):
  out = []
  for word in words:
    if word == "," or word == ";" or word == ".":
      out.append(',')
    elif len(word) == 0 or string.punctuation.find(word[0]) > -1:
      pass
    elif word.find("-"):
      subwords = word.split("-")
      for sub in subwords:
        out.append(sub)
    else:
      out.append(word)
  out2 = []
  for word in out:
    if word.endswith("."):
      out2.append(word[:-1])
      out2.append(",")
    else:
      out2.append(word)
  return out2

# inverse of above, unhyphenated words that might have both words
#def hyphen(words, worddict):
#  out = []
#  dic = pyphen.Pyphen(lang='en_US')
#  #for (x, y) in dic.iterate('hoofprint'):
#  #  print(x)
#  #  print(y)
#  for word in words:
#    #print(word)
#    if worddict.get(word, None) != None:
#      #print("ok")
#      out.append(word)
#    else:
#      finished = False
#      for (first,second) in dic.iterate(word):
#        if worddict.get(first, None) != None and worddict.get(second, None) != None and not finished:
#          #print(first + "," + second)
#          out.append(first)
#          out.append(second)
#          finished = True
#      if not finished:
#        return []
#  print(out)
#  return out

def concat(toks):
  out = ""
  for t in toks:
    out += t
  return out
      
def hyphen(words, worddict):
  out = []
  for word in words:
    if word in worddict:
      out.append(word)
    else:
      toks = hyphenate_word(word)
      split = False
      for i in range(1, len(toks)):
        s1 = concat(toks[0:i])
        s2 = concat(toks[i:len(toks)])
        if worddict.get(s1, None) != None and worddict.get(s2, None) != None:
          out.append(s1)
          out.append(s2)
          split = True
      if not split:
        out.append(word)
  return out
      

#print(hyphen(['wordset'], {}))
#print(hyphen(['wordset'], {'word':0, 'set':1}))
#print("done")
#
## return array of tokenized wordsets for a number
#def digitize(number):
#  pass
#
#print(hyphen(['hoofprint'], {'hoof': ['H', 'OOF'], 'print': ['PR INT']}))
#print(hyphen(['blunderbuss'], {}))
#      
if __name__ == "__main__":
  print(tokenize("the monkeys, they hate me,"))
  #print(tokenize2("the monkeys, they hate me,"))
