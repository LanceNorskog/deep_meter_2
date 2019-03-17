import sys
import ast
  
# official one-hot dictionary for ARPAbets
# removed stress indicators, they're implied and can be ? for one-syllable words
  
stop_arpabet = ','
pause_arpabet = ','
stop_encoding = 0
pause_encoding = 1
  
class arpabets:
  def __init__(self):
    self.arpabets = ['.', ',', 'AH', 'N', 'IH', 'T', 'D', 'R', 'S', 'L', 'DH', 'AE', 'Z', 'M', 'ER', 'EH', 'IY', 'F', 'K', 'V', 'W', 'P', 'HH', 'AO', 'B', 'EY', 'AY', 'UW', 'AA', 'NG', 'OW', 'G', 'SH', 'AW', 'TH', 'CH', 'JH', 'Y', 'UH', 'OY', 'ZH']
    self.counts = [0, 0, 11322, 9097, 7450, 7444, 7129, 6447, 6048, 5471, 5136, 4645, 3980, 3305, 2968, 2940, 2844, 2740, 2715, 2621, 2540, 2295, 2287, 2283, 2058, 2026, 1956, 1943, 1913, 1528, 1452, 1153, 752, 745, 669, 583, 566, 447, 324, 189, 55]
    self.num_arpabets = len(self.arpabets)
    self.encodings = {}
    for i in range(self.num_arpabets):
      self.encodings[self.arpabets[i]] = i

  def get_size(self):
    return self.num_arpabets
  
  def get_encoding(self, syll):
    return self.encodings.get(syll)

  def get_counts(self):
    return self.counts
  
  def get_arpabet(self, encode):
    if encode < self.num_arpabets:
      return self.arpabets[encode]
    # NOT SUPPOSED TO HAPPEN
    return -1

  # debug helper for notebook
  def interpret(self, pred, valid=0.5):
    found = []
    arpas = []
    for i in range(len(pred)):
      enc = pred[i]
      if enc >= valid:
        found.append(i)
        arpas.append(self.get_arpabet(int(i % self.num_arpabets)))
    #print(found)
    #print(arpas)

  def interpret2(self, pred):
    found = []
    arpas = []
    i = 0
    while i * self.num_arpabets < len(pred):
      max_j = -1 
      max_d = -10000000
      for j in range(self.num_arpabets):
        v = pred[i * self.num_arpabets + j]
        #print('{0}, {1} = {2}'.format(i, j, v))
        if v > max_d:
          max_d = v
          max_j = j
      #print('{0}, {1}'.format(max_j, max_d))
      found.append(max_j)
      arpas.append(self.get_arpabet(max_j))
      i += 1
    return (found, arpas)
  
if __name__ == "__main__":
  s = arpabets()
  print(s.get_size())
  print(s.get_encoding('.'))
  print(s.get_encoding(','))
  print(s.get_encoding('EY'))
  print(s.get_arpabet(47))
  print(s.counts[4])

