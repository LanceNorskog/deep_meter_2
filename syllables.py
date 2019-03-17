import sys
import ast
  
# official one-hot dictionary- pick a smaller number, index from beginning
  
unknown_syllable = '?'
pause_syllble = ','
unknown_encoding = 0
pause_encoding = 1
  
class syllables:
  def __init__(self, size=100000):
    f = open("blobs/allsyllables.array")
    self.syllables = ast.literal_eval(f.readline())
    self.counts = ast.literal_eval(f.readline())
    if size < len(self.syllables):
      self.syllables = self.syllables[0:size]
    self.num_syllables = len(self.syllables)
    self.encodings = {}
    for i in range(self.num_syllables):
      self.encodings[self.syllables[i]] = i

  def get_size(self):
    return self.num_syllables
  
  def get_encoding(self, syll):
    return self.encodings.get(syll, unknown_encoding)

  def get_counts(self):
    return self.counts
  
  def get_syllable(self, encode):
    if encode < self.num_syllables:
      return self.syllables[encode]
    return unknown_syllable

  # debug helper for notebook
  def interpret(self, pred, valid=0.5):
    found = []
    sylls = []
    for i in range(len(pred)):
      enc = pred[i]
      if enc >= valid:
        found.append(i)
        sylls.append(self.get_syllable(int(i % self.num_syllables)))
    print(found)
    print(sylls)

  def interpret2(self, pred):
    found = []
    sylls = []
    i = 0
    while i * self.num_syllables < len(pred):
      max_j = 0 # unknown
      max_d = -10 # impossible
      for j in range(self.num_syllables):
        v = pred[i * self.num_syllables + j]
        #print('{0}, {1} = {2}'.format(i, j, v))
        if v > max_d:
          max_d = v
          max_j = j
      #print('{0}, {1}'.format(max_j, max_d))
      found.append(max_j)
      sylls.append(self.get_syllable(max_j))
      i += 1
    print(found)
    print(sylls)
  
if __name__ == "__main__":
  s = syllables()
  print(s.get_size())
  print(s.get_encoding('?'))
  print(s.get_encoding(','))
  print(s.get_encoding('EY'))
  print(s.get_syllable(47))
  print(s.counts[4])
  s.num_syllables = 4
  print('--------')
  s.interpret2([0, 0, 0.5, 0.4])

