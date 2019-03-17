
import sys
import numpy as np

# sort lines by hash, its a good randomizer that lets us find dups
hashes = []
text = []

for line in sys.stdin:
  hashes = hashes + [hash(line)]
  text.append(line)

#print(hashes[:200])

indices = np.argsort(hashes)
#print(indices[:200])
  
out = open("data.dev", "w")
j = 1 
last = ''
for i in indices:
  if text[i] != last:
    out.write(text[i])
  if j == 5000:
    out.close()
    out = open("data.test", "w")
  if j == 10000:
    out.close()
    out = open("data.train", "w")
  j += 1
  last = text[i]

out.close()
