
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

if __name__ == "__main__":
    preds = np.array([[[0, 1, 2], [4, 3, 5], [8,6,7]]])
    print('predictions: ', preds)
    (v, i) = topk(preds, top_k=1)
    print('vals:  ', v)
    print('inds: ', i)
            
    
