# Manipulation of Penn Treebank constituency parse trees
# Keyed to output of Allennlp Constituency parser

from nltk import Tree
import snlp

clause_labs = {
    'FRAG': None,
    'NP': None,
    'PP': None,
    'S': None,
    'SBAR': None,
    'SBARQ': None,
    'SINV': None,
    'SQ': None,
    'VP': None,
    'WHNP': None,
    'X': None
}

def clauses(t):
    for st in t.subtrees():
         if st.label().upper() in clause_labs and len(st) > 1:
             yield st

# Return a tree with any and all instances of X removed
# This is the core of summarization: remove nonessential parts of the parse tree!
# Another rewriter might be synonym substitution using PPDB.

def trim_ADJP(t):
    for st in snlp.strip(t, ['ADJP', 'adjp']):
        yield st

def trim_ADVP(t):
    for st in snlp.strip(t, ['ADVP', 'advp']):
        yield st

def trim_WHNP(t):
    for st in snlp.strip(t, ['WHNP', 'whnp']):
        yield st

rewriters = [ trim_ADJP, trim_ADVP, trim_WHNP ]

# return pair of (clause, possibly trimmed version of clause)
def pairs(t):
    for clause in clauses(t):
        yield (clause, clause)
        for rewriter in rewriters:
            for st in rewriter(clause):
                yield (clause, st)

def emit(words):
    punc = {'.':None, ',':None, ';':None, ':': None, '?':None, "'s":None}
    sent = ''
    start = 0
    if words[0] in punc:
        start = 1
    for i in range(start, len(words) - 1):
        sent += words[i]
        if not words[i + 1] in punc:
            sent += ' '
    sent += words[-1]
    return sent

def sentence(t):
    s = str(t.flatten())
    s = s[1:]
    s = s[:-1]
    words = s.split()
    key = words[0]
    return (key, emit(words[1:]))
    
if __name__ == "__main__":
    t = Tree.fromstring("(NP (NP (DT A) (NN bathroom)) (PP (IN with) (NP (NP (NNS walls)) (SBAR (WHNP (WDT that)) (S (VP (VBP are) (VP (VBN painted) (ADJP (NN baby) (JJ blue)))))))) (. .))")

    print('-------------------------------------------')
    print(sentence(t))
    print('clauses, rewriters')
       
    for c in clauses(t):
        print(c)
        #for rewriter in rewriters:
        #    for st in rewriter(t):
        #        print(st)
        #        print()
        #print()

    print('Trimmed')

    out = {}
    for (clause, trimmed) in pairs(t):
        #print('# {}'.format(clause))
        #print('> {}'.format(trimmed))
        x = str(clause.flatten()) + ' -> \n\t' + str(trimmed.flatten())
        out[x] = x

    for k in out.keys():
        print(k)
