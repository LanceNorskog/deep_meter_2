
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

def decodem(scorepaths, decoder, wordset):
    morepaths = np.zeros(scorepaths.shape, dtype='int32')
    for j in range(scorepaths.shape[0]):
        #print('scorepaths[{}]: {}'.format(j, scorepaths[j]))
        #print('top_paths.shape: ', top_paths.shape)
        #print('top_paths[{}]: {}'.format(j, top_paths))
        #print('top_paths[{}][]: {}'.format(j, top_paths[0][np.arange(num_sylls), scorepaths[j]]))
        morepaths[j] = top_paths[np.arange(num_sylls), scorepaths[j]]
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

    from mappers import Decoder
    from full import FullSearch

    syllables = {'therefore':['DH EH R', 'F AO R'], 'the':['DH AH'], 'mugger':['M AH', 'G ER'], 'is': ['IH Z'], 'there':['DH EH R'], 'for':['F AO R'], 'me':['M IY']}
    decoder = Decoder(syllables)
    top_k = 9
    num_sylls = 5
    num_dict = 15
    fs = FullSearch(num_sylls * 2, num_sylls, top_k)
    (top_vals, top_paths) = get_top_k(np.zeros((num_sylls, num_dict), dtype='float32'), top_k=top_k)
    fs.mainloop(top_paths)
    sentences = decodem(fs.scorepaths, decoder, [])
    
