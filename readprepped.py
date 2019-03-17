# Read prepped data in format:  text tab [syllables ]

import numpy as np
from ast import literal_eval
import syllables
import utils

# read classified poetry lines: text tab [['syll', 'la', 'ble'], ...]
# clip to only most common syllables with syllable manager
# ['words', ...], [[[0,0,1,0], ...]]
def get_data(filename, arpabet_mgr, num_symbols, max_lines=10000000000):
    stop_arpabet = 0
    num_arpabets = arpabet_mgr.get_size()      
    lines = open(filename, 'r').read().splitlines()
    text_lines = []
    text_arpabets = []
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    for i in range(0, len(lines)):
      if i == num_lines:
        break
      parts = lines[i].split("\t")
      syllables = literal_eval(parts[1])
      #print(syllables)
      arpas = []
      for s in syllables:
        for p in s:
          for x in p.split(' '):
            arpas.append(x)
      #print(arpas)
      if len(arpas) < num_symbols:
        text_lines.append(str(parts[0]))
        text_arpabets.append(arpas)
    num_lines = len(text_lines)
    label_array = np.zeros((num_symbols, num_lines, num_arpabets), dtype=np.int8)
    for i in range(0, num_lines):
      for j in range(num_symbols):
        label_array[j][i][stop_arpabet] = 1
        # variable-length list of syllables
        if j < len(text_arpabets[i]):
          enc = arpabet_mgr.get_encoding(text_arpabets[i][j])
          if enc >= 0 and enc < num_arpabets:
            label_array[j][i][enc] = 1
            label_array[j][i][stop_arpabet] = 0
    return (text_lines, label_array)

# read classified poetry lines: text tab [['syll', 'la', 'ble'], ...]
# clip to only most common syllables with syllable manager
# ['words', ...], [[['DH','AH'], ...]]
def read_prepped(filename, syll_mgr, num_symbols, max_lines=1000000):
    num_syllables = syll_mgr.get_size()      
    lines = open(filename, 'r').read().splitlines()
    num_lines = min(max_lines, len(lines))
    text_lines = []
    text_sylls = []
    for i in range(num_lines):
      parts = lines[i].split("\t")
      syll_array = literal_eval(parts[1])
      if len(utils.flatten(syll_array)) == num_symbols:
        text_lines.append(str(parts[0]))
        text_sylls.append(syll_array)
    return (text_lines, text_sylls)

# given [[['DH', 'AE'] ...] ,...] return numeric labels, separate array 
# return [ num_symbols ][ num_lines ][ one_hot of syllable encoding ]
def get_onehot_multi(text_sylls, syll_mgr, num_symbols):
    num_lines = len(text_sylls)
    num_syllables = syll_mgr.get_size()
    onehot_array = np.zeros((num_symbols, num_lines, num_syllables), dtype=np.int8)
    for i in range(num_lines):
      data = utils.flatten(text_sylls[i])
      for j in range(num_symbols):
        onehot_array[j][i][syll_mgr.get_encoding(data[j])] = 1
    return onehot_array
     

if __name__ == "__main__":
    syll_mgr = syllables.syllables()
    (text, sylls) = read_prepped('prepped_data/gutenberg.iambic_pentameter', syll_mgr, 10)
    print("Read {} lines of text".format(len(text)))
    print("{} -> {}".format(text[0], sylls[0]))
    onehots = get_onehot_multi(sylls, syll_mgr, 10)
    print("Shape: {}".format(onehots.shape))
    print("{}[0] -> {}".format(text[0], onehots[0][0]))
