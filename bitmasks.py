# manage set of 63-position bitmasks, one per phoneme in a sentence
# each entry is a list of masks. create cross-product of all possible lists, isolate lists that have no mutual coverage

from itertools import combinations

class sentence:

  def __init__(self, size):
    self.size = size
    self.fullmask = 0
    for i in range(size):
      self.fullmask |= 1 << i

  def clear(self):
    self.row = {}

  def set(self, key, first, last):
    def getmask(first, last):
      i = first
      x = 1 << first
      while i < last:
        x |= 1 << i
        i += 1
    mask = getmask(first, last)
    self.row[key] = mask

  def check(self, min, max):
    combo_list = []
    for n in range(min, max + 1):
      for combo in combinations(self.row, n):
        full = 0
        for mask in combo:
          if full & mask:
            break
        if 
