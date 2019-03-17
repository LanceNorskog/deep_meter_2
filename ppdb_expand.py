#!/usr/bin/python

#
# Expand sentences based on synonym database from PPDB
#

from ppdb.ppdb_cmudict import load_ppdb as load_ppdb
from itertools import product

ppdb = load_ppdb('/home/lance/open/data/ppdb-2.0-xxxl.pk')

# rotate through possible sentences using ppdb synonyms
def expand(words):
    l = len(words)
    pos = 0
    h = hash(' '.join(words))
    off = [0] * l
    for i in range(l):
        off[i] = len(ppdb[words[i]][0])
    out = [[]] * l
    for i in range(l):
        out[i] = [words[i]]
        more = ppdb[words[i]][0]
        for pair in more:
            w = pair[0]
            out[i].extend([w])
    for s in product(*out):
        yield s
    

if __name__ == '__main__':
    for x in expand(['these', 'are', 'accolades']):
        print(x)
