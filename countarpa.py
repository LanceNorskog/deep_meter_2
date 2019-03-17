import collections
import ast
import sys
import utils

count = collections.Counter()

total = 0
sum = 0
min = 10000000
max = -min
for line in sys.stdin:
  text = line.split("\t")[1]
  symbols = ast.literal_eval(text)
  arpas = []
  for s in symbols:
    for p in s:
      for x in p.split(' '):
        arpas.append(x)
  sum += len(arpas)
  total += 1
  if min > len(arpas):
    min = len(arpas)
  if max < len(arpas):
    max = len(arpas)
  for a in arpas:
    count[a] += 1

print("Total {0}, mean {1}, max {2}, min {3} ".format(total, str(sum / total), max, min))

# stop, pause are special syllables #0 and #1
sorted = count.most_common(len(count))
arpa_array = ['.', ',']
for (key,val) in sorted:
  arpa_array.append(key)
print(str(arpa_array))
arpa_count = [0, 0]
for (key,val) in sorted:
  arpa_count.append(val)
print(str(arpa_count))
