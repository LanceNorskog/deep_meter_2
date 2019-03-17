# haiku gen 1 column

import sys

import snlp
import cmudict

nsyll=5

cd = cmudict.CMUDict()

# TODO
# looking good except for sentences with hyphenated words.
# they don't get clipped, just keep going ??




num_lines=0
num_pass=0
num_nonword=0

outf_5 = open("haiku_5.txt", "w")
outf_7 = open("haiku_7.txt", "w")
num_5=0
num_7=0

lastline=''
for line in sys.stdin:
    if line == lastline:
        continue
    lastline = line
    line = line[:-1].lower()
    if line[0] != '(':
        continue
    t = snlp.parse(line)
    # t = snlp.strip(t, ['``', '.', '"', '"'])
    # max words can include punctuation
    for sample in snlp.clauses(t, _min=2, _max=8, _minlen=10):
        sylls = 0
        badword = None
        sample = snlp.punct(sample)
        for word in sample:
            if word in ["'", ',', '.', '``']:
                continue
            if word in cd.stress_dict:
                sylls += len(cd.stress_dict[word])
            else:
                num_nonword += 1
                break
        if sylls == 5:
            outf_5.write(' '.join(sample) + '\n');
            num_5 += 1
        elif sylls == 7:
            outf_7.write(' '.join(sample) + '\n');
            num_7 += 1
    num_lines += 1

outf_5.close()
outf_7.close()
print('Found 5-sylls: {}, 7-sylls: {}'.format(num_5, num_7))
print('Lines: {}, clauses rejected: {}'.format(num_lines, num_nonword))
snlp.stats()
