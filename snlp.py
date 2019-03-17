# read S-tree format from nltk.parse.stanford.GenericStanfordParser.parse_sents() method
# really just MultiNLI data
# generate certain sentence or clause variations

from nltk.tree import Tree
import itertools

# counters
num_clause = 0
num_accepted = 0
num_combo = 0


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

# break sentence
def parse(sentence):
    return Tree.fromstring(sentence)

def clauses(t, filter=None, _min=0, _max=1000, _minlen=1):
    global num_clause
    global num_accepted
    out = {}
    for t2 in t.subtrees(filter):
        t2 = flatten(t2)
        num_clause += len(t2)
        if len(t2) >= _min and len(t2) <= _max:
            str = ' '.join(t2)
            if len(str) >= _minlen:
                out[str] = t2 
    num_accepted += len(out.keys())
    return list(out.values())

def flatten(t):
    out = []
    for x in t.flatten():
        out.append(x)
    return out

def punct(words):
    # hanging possessive problem with SNLI corpus
    if words[0] == "'" or words[0] == "'s":
        words = words[1:]
    # cmudict stores possessives
    out = []
    i = 0
    while i < len(words) -1:
        if words[i + 1] == "'s":
            out += [ words[i] + "'s" ]
            i += 1
        else:
            out += [ words[i] ]
        i += 1
    words = out
    return words
    

# sample = '(ROOT (S (NP (DT This) (NN site)) (VP (VBZ includes) (NP (NP (NP (DT a) (NN list)) (PP (IN of) (NP (DT all) (NN award) (NNS winners)))) (CC and) (NP (NP (DT a) (JJ searchable) (NN database)) (PP (IN of) (NP (NNP Government) (NNP Executive) (NNS articles)))))) (. .)))'
sample = '(ROOT (S (NP (PRP I)) (VP (VP (VBP like) (NP (PRP him)) (PP (IN for) (NP (DT the) (JJS most) (NN part)))) (, ,) (CC but) (VP (MD would) (ADVP (RB still)) (VP (VB enjoy) (S (VP (VBG seeing) (S (NP (NN someone)) (VP (VB beat) (NP (PRP him))))))))) (. .)))'

def filter_f(t):
    print("'" + t.label() + "'")
    return True

def filter_RB(t):
    return t.label() != 'ADVP'

def strip(t, labs):
    treeType = type(t)
    def walk(t):
        i = len(t) - 1
        while i > -1:
            if type(t[i]) == treeType and t[i].label() in labs:
                # print('removing {} from t ({})'.format(i, t, t[i]))
                t.pop(i)
                i = len(t) - 1
            else:
                i -= 1
        for sub in t:
            if type(sub) == treeType:
                walk(sub)

    def walk2(t):
        bad = []
        for i in range(len(t) - 1, 0, -1):
            if type(t[i]) == treeType and t[i].label() in labs:
                print('removing {} from t ({})'.format(i, t, t[i]))
                bad += [i]
        for i in bad:
            print('Remove {} from {}'.format(t[i], t))
            t.pop(i)
        for sub in t:
            if type(sub) == treeType:
                walk2(sub)
                
    t = t.copy(deep=True)
    walk(t)
    return t

# given a parse tree, return all subtrees with any and all POS label subtrees removed
def combos(t, labs):
    global num_combo
    out = {}
    for lab in list(powerset(labs)):
        print(list(lab))
        t2 = strip(t, list(lab))
        for x in clauses(t2, _min=1, _max=7, _minlen=10):
            out[str(x)] = x
    num_combo += len(out.keys())
    return out.keys()

# print stats
def stats():
    print('Possible clauses: {}, unique clauses: {}, combos: {}'.format(num_clause, num_accepted, num_combo))

# various tests
if __name__ == '__main__':
    print(punct(["'s", 'start', 'middle']))
    print(punct(['start', "'s"]))
    print(punct(['start', "'s", 'end']))
    print(punct(['start', "'s", 'end', "'s"]))
    
    t = Tree.fromstring(sample)
    print('all:')
    for x in clauses(t, _min=3, _max=10):
        print(x)
    print('\nno adverbs:')
    t2 = strip(t, ['PP'])
    for x in clauses(t2, filter=filter_RB, _min=2, _max=5, _minlen=10):
        print(x)
    print('\ncombos with various words ripped')
    for c in combos(t, ['RB', 'PP']):
        print(c)
    stats()
