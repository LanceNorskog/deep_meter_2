
import numpy as np

''' return (summed weights, syllable indices) from prediction set '''
def topk(predictions, top_k=5):
    vals = np.zeros((predictions.shape[0], predictions.shape[1], top_k), dtype='float32')
    indices = np.zeros((predictions.shape[0], predictions.shape[1], top_k), dtype='int32')
    for p in range(len(predictions)):
        for s in range(len(predictions[0])):
            ind = np.argsort(predictions[p][s])
            short = np.flip(ind[-top_k:])
            vals[p][s] = predictions[p][s][short]
            indices[p][s] = short
    return (vals, indices)

''' return the original indices that are chosen '''
def topk_lookup(vals, indices, chosen):
    #print('vals:  ', vals)
    #print('inds: ', indices)
    #print('chosen: ', chosen)
    return (vals[np.arange(len(vals)), chosen], indices[np.arange(len(vals)), chosen])

if __name__ == "__main__":
    preds = np.array([[[0, 1, 2], [4, 3, 5], [8,6,7]]])
    print('predictions: ', preds)
    (v, i) = topk(preds, top_k=2)
    print('preds: ', preds)
    (rv, ri) = topk_lookup(v[0], i[0], np.array([[0, 0, 0],[1,1,1]]))
    print('rv: ', rv)
    print('ri: ', ri)
            
    
