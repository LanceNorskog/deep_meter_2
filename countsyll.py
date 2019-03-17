import collections
import ast
import sys

count = collections.Counter()

for line in sys.stdin:
  text = line.split("\t")[1]
  sylls = ast.literal_eval(text)
  for syll in sylls:
    for s in syll:
      count[s] += 1

#print(count.most_common(200000))

#commons = count.most_common(2000)

#print("length " + str(len(count)))
#print(commons[0])
#print(commons[-1])
  
#print(commons)
# unknown, pause are special syllables #0 and #1
sorted = count.most_common(len(count))
syll_array = ['?', ',']
for (key,val) in sorted:
  syll_array.append(key)
print(str(syll_array))
syll_count = [0, 0]
for (key,val) in sorted:
  syll_count.append(val)
print(str(syll_count))
