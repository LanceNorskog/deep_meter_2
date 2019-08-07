
import numpy as np

''' return (summed weights, syllable indices) from 1 prediction set '''
def get_top_k(predictions, top_k=5):
    print(predictions.shape)
    assert len(predictions.shape) == 2

    vals = np.zeros((predictions.shape[0], top_k), dtype='float32')
    indices = np.zeros((predictions.shape[0], top_k), dtype='int32')
    for s in range(predictions.shape[0]):
        ind = np.argsort(predictions[s])
        short = np.flip(ind[-top_k:])
        vals[s] = predictions[s][short]
        indices[s] = short
    return (vals, indices)

def decodewords(scorepaths, top_paths, idx2word):
    morepaths = np.zeros(scorepaths.shape, dtype='int32')
    for j in range(scorepaths.shape[0]):
        #print('scorepaths[{}]: {}'.format(j, scorepaths[j]))
        #print('top_paths.shape: ', top_paths.shape)
        #print('top_paths[{}]: {}'.format(j, top_paths))
        #print('top_paths[{}][]: {}'.format(j, top_paths[0][np.arange(num_sylls), scorepaths[j]]))
        morepaths[j] = top_paths[np.arange(top_paths.shape[0]), scorepaths[j]]
    sentences = {}
    for j in range(scorepaths.shape[0]):
        words = []
        for k in range(scorepaths.shape[1]):
            words.append(idx2word[morepaths[j][k]])
        sentence = ' '.join(words)
        sentences[sentence] = words
    return sentences

def decodem(scorepaths, top_paths, decoder, wordset, wordmap):
    morepaths = np.zeros(scorepaths.shape, dtype='int32')
    for j in range(scorepaths.shape[0]):
        #print('scorepaths[{}]: {}'.format(j, scorepaths[j]))
        #print('top_paths.shape: ', top_paths.shape)
        #print('top_paths[{}]: {}'.format(j, top_paths))
        #print('top_paths[{}][]: {}'.format(j, top_paths[0][np.arange(num_sylls), scorepaths[j]]))
        morepaths[j] = top_paths[np.arange(top_paths.shape[0]), scorepaths[j]]
    #print('morepaths: ' + str(morepaths))
    encoded = decoder.get_sentences(morepaths)
    sentences = {}
    if len(encoded) > 0:
        #print(encoded)
        decoded = []
        for e1 in encoded:
            if len(e1) > 0 and len(e1[0]) > 0:
                dec = decoder.decode_sentences([e1])
                decoded.append(dec)
        for d1 in decoded:
            for d2 in d1:
                for d3 in d2:
                    for d4 in d3:
                        go = True
                        _lastidx = -1
                        for w in d4:
                            if not w in wordset:
                                go = False
                            _idx = decoder.word2idx[w]
                            if _lastidx > 0:
                                if _lastidx == _idx or not wordmap.get(_lastidx, _idx):
                                    go = False
                                    #print('Fail: {},{} {},{}'.format(_lastidx, _idx, _lastword, w))
                            _lastidx = _idx
                            _lastword = w
                        if go:
                            key = ' '.join(d4)
                            sentences[key] = d4
                    #print('d3: ', d3)
                    #key = ' '.join(d3)
                    #sentences[key] = d3
    return sentences

# return N possible sentences with the fewest words
def short_sentences(sentences, num_sylls):
    out = {}
    for i in range(1, num_sylls + 1):
        for (k, v) in sentences.items():
            if len(v) == i:
                out[k] = v
        if len(out) > 4:
            return out
    return out

if __name__ == "__main__":
    preds = np.array([[0, 1, 2], [4, 3, 5], [8,6,7]])
    print('predictions: ', preds)
    (v, i) = get_top_k(preds, top_k=2)
    print('vals:  ', v)
    print('inds: ', i)

    from syllables_cmu import syllables
    from mappers import Decoder
    from full import FullSearch
    from wordmap import Wordmap

    decoder = Decoder(syllables)

    idx2word = [''] * 50
    for i in range(50):
        idx2word[i] = str(i)
    

    top_k = 2
    num_sylls = 5
    num_dict = 15
    fs = FullSearch(num_sylls * 2, num_sylls, top_k)
    (top_vals, top_paths) = get_top_k(np.zeros((num_sylls, num_dict), dtype='float32'), top_k=top_k)
    print('top_paths.shape ', top_paths.shape)
    fs.mainloop(top_paths)
    haikuwordset = set()
    wordmap = Wordmap(100)
    # sentences = decodem(fs.scorepaths, top_paths, decoder, haikuwordset, wordmap)
    sentences = decodewords(fs.scorepaths, top_paths, idx2word)
    print('Sentences:')
    for s in sentences.keys():
        print('... ', s)
