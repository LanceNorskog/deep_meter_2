# haiku gen 1 column

import sys

import snlp
import cmudict

cd = cmudict.CMUDict()

# Find single lines of Haiku inside Constituency Parse output.
# Search for 5-syllable and 7-syllable lines
# Stage 1 (basic):
#    Generate possible subtrees, including those missing certain CP tags.
#    DONE
# Stage 2 (entailment):
#    Include all trees that are parent of a sample.
#    Emit output that maps all parent trees to a sample.
#    Allows training encoder->decoder with entailed output.
#    That is, the output is a summary of the input, instead of just a direct copy.
#    This gives a lot more flexibility in creating a predictor.


num_lines=0
num_pass=0
num_nonword=0

outf_5 = open("haiku_5.txt", "w")
outf_7 = open("haiku_7.txt", "w")
num_5=0
num_7=0

drop_clauses=['adjp','whnp','s','sbar','sbarq','sinv','sq']

def process(sample):
        # print(sample)
        sylls = 0
        badword = False
        # sample = snlp.punct(sample)
        for word in sample:
            if word in ["'", ',', '.', '``']:
                continue
            if word in cd.stress_dict.keys():
                sylls += len(cd.stress_dict[word])
            else:
                num_nonword += 1
                badword = True
                print('bad: {}'.format(word))
                break
        if badword:
            continue
        if sylls == 5:
            outf_5.write(' '.join(sample) + '\n');
            num_5 += 1
        elif sylls == 7:
            outf_7.write(' '.join(sample) + '\n');
            num_7 += 1


for line in sys.stdin:
    if line[0] != '(':
        continue
    line = line[:-1].lower()
    t = snlp.parse(line)
    t = snlp.strip(t, ['``', '.', '"', '"'])
    # max words can include punctuation
    # for sample in snlp.combos(t, labs=drop_clauses):
    # for sample in snlp.clauses(t, _min=1, _max=7, _minlen=10):
    for sample in snlp.clipped_unique(t, drop_clauses):
        process(sample)
    num_lines += 1

outf_5.close()
outf_7.close()
print('Found 5-sylls: {}, 7-sylls: {}'.format(num_5, num_7))
print('Lines: {}, clauses rejected: {}'.format(num_lines, num_nonword))
snlp.stats()

