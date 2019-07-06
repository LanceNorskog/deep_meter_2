
def get_mappers(word2sylls):
    big_sylls = set()
    for index, word in enumerate(word2sylls):
        sylls = word2sylls[word]
        for syll in sylls:
            syll = syll.replace(' ', '').lower()
            big_sylls.add(syll)

    syll2idx = {}
    for index, syll in enumerate(big_sylls):
        syll2idx[syll] = index
    idx2syll = [0] * len(syll2idx)
    for index, syll in enumerate(syll2idx):
        idx2syll[index] = syll
    
    return (syll2idx, idx2syll)


if __name__ == "__main__":
    syllables = {'the':['DH AH']}
    (syll2idx, idx2syll) = get_mappers(syllables)
    print(syll2idx.keys())
    print(syll2idx['dhah'], idx2syll[0])
    print('# features: ', len(idx2syll))


