import argparse
import ast
import syllables
import sys

# find subspace of dataset of most common lines
#   load syllables of lines
#   create line-syllable matrix from master syllable encodings as columns
#   pick "most normal" rows from dot-product
#   reload file and print those

parser = argparse.ArgumentParser()
parser.add_argument('--file', help='file', required=True)
args = parser.parse_args()

syllables.set_size(500)
sys.stderr.write("current size = " + str(syllables.get_size()) + "\n")

i = 0
for line in open(args.file, "r"):
  sylls = ast.literal_eval(line.split("\t")[1])
  #print(sylls)
  i += 1
  if i % 100000 == 0:
    sys.stderr.write(str(i) + "...\n")
  fail = False
  for sylla in sylls:
    for syll in sylla:
      enc = syllables.get_encoding(syll)
      if enc == syllables.unknown_encode:
        #sys.stderr.write(syll + "\n")
        fail = True
        break
    if fail:
      break
  if not fail:
    print(line[0:-1]) # trim newline


