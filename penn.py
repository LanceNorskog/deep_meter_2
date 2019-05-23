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
         if st.label() in clause_labs and len(st) > 2:
             yield st

# Return a tree with any and all instances of X removed

def trim_ADJP(t):
    for st in snlp.strip(t, ['ADJP']):
        yield st

def trim_ADVP(t):
    for st in snlp.strip(t, ['ADVP']):
        yield st

def trim_WHNP(t):
    for st in snlp.strip(t, ['WHNP']):
        yield st

rewriters = [ trim_ADJP, trim_ADVP, trim_WHNP ]

def pairs(t):
    #print('tree ' + str(t))
    for clause in clauses(t):
        #print('clause ' + str(t))
        for rewriter in rewriters:
            for st in rewriter(clause):
                yield (clause, st)
    

if __name__ == "__main__":
    t = Tree.fromstring("(NP (NP (DT A) (NN bathroom)) (PP (IN with) (NP (NP (NNS walls)) (SBAR (WHNP (WDT that)) (S (VP (VBP are) (VP (VBN painted) (ADJP (NN baby) (JJ blue)))))))) (. .))")

    print('-------------------------------------------')
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
        x = str(clause) + ' -> \n\t' + str(trimmed)
        out[str(x)] = x

    for k in out.keys():
        print(k)
