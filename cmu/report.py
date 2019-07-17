
# per prediction, not multiple predictions

# fetch best predictions per syllable
def find_top_k_match(data, prediction, top_k=5):
        
        out = [-1] * len(data)
        for i in range(len(data)):
            topind = np.argsort(prediction[i])
            topind = topind[-top_k:]
            for j in range(top_k):
                #print(data[i][0], topind[j])
                if data[i][0] == topind[j]:
                    out[i] = topind[j]
        return out
    
# Independent reimplementation of top5 sparse and top5 perfect, since can't get a good top-5 perfect metric
def report(data, prediction):
    def match(data, prediction):
        assert len(data.shape) == 2
        assert len(prediction.shape) == 2
        good = 0
        top5 = 0
        count = 0
        for i in range(len(data)):
            topind = np.argsort(prediction[i])
            if data[i][0] == topind[-1]:
                good += 1
            topind = topind[-5:len(topind)]
            for j in range(5):
                if data[i][0] == topind[j]:
                    top5 += 1
                    break
            count += 1
        return (good, top5, count)

    _sparse = 0.0
    _perfect = 0.0
    _sparse5 = 0.0
    _perfect5 = 0.0
    _total = 0
    for n in range(len(data)):
        #print(len(short[n]))
        (good, top5, count) = match(data[n], predicts[n])
        if count == 0:
            continue
        _sparse += good/count
        _sparse5 += top5/count
        if good == count:
            _perfect += 1  
        if top5 == count:
            _perfect5 += 1
        _total += 1
    return {'sparse':_sparse/_total, 'perfect': _perfect/_total, 'sparse5': _sparse5/_total, 'perfect5': _perfect5/_total}

