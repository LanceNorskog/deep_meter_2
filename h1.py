# haiku gen 1 column

import sys

import snlp
import summarize
import cmudict

cd = cmudict.CMUDict()

# Find single lines of Haiku inside Constituency Parse output.
# Search for 5-syllable and 7-syllable lines
# Stage 1 (basic):
#    Generate possible subtrees, including those missing certain CP tags.
#    DONE
# Stage 2 (clause summarization):
#    Find subsentences/clauses that can be validly trimmed by dropping one or more labeled subtrees.
#    Allows training encoder->decoder with summarized output.
#    That is, the output is a summary of the input, instead of just a direct copy.
#    This gives a lot more flexibility in creating a predictor.


num_lines=0
num_nosyll=0

nosyll_f = open("no_syll.txt", "w")
outf_5 = open("haiku_5.txt", "w")
outf_7 = open("haiku_7.txt", "w")
num_5=0
num_7=0
last_nosyll = None

drop_clauses=['adjp','whnp','s','sbar','sbarq','sinv','sq']

# accept two clauses, input generates output
def process(input, output):
        global num_nosyll
        global last_nosyll
        global num_5
        global num_7
        sylls = 0
        nosyll = False
        # sample = snlp.punct(sample)
        for word in output:
            if word in ["'", ',', '.', '`']:
                continue
            if word in cd.stress_dict.keys():
                sylls += len(cd.stress_dict[word])
            else:
                num_nosyll += 1
                nosyll = True
                if word != last_nosyll:
                    nosyll_f.write(word + '\n')
                    last_nosyll = word
                break
        if nosyll:
            return
        if sylls == 5:
            outf_5.write(summarize.emit(input) + '\t' + summarize.emit(output) + '\n');
            num_5 += 1
        elif sylls == 7:
            outf_7.write(summarize.emit(input) + '\t' + summarize.emit(output) + '\n');
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
    samples = {}
    for sample in snlp.clipped_unique(t, drop_clauses):
        samples[str(sample) + '->' + str(sample)] = (sample, sample)
    for (inp_t, outp_t) in summarize.pairs(t):
        # print('{} -> {}'.format(inp, outp))
        samples[str(inp_t) + '->' + str(outp_t)] = (snlp.flatten(inp_t), snlp.flatten(outp_t))
    for (inp_t, outp_t) in samples.values():
        process(inp_t, outp_t)

    num_lines += 1

outf_5.close()
outf_7.close()
print('Found 5-sylls: {}, 7-sylls: {}'.format(num_5, num_7))
print('Lines: {}, clauses rejected for non-CMU word: {}'.format(num_lines, num_nosyll))
snlp.stats()
