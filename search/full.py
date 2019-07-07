import numpy as np

from itertools import product

class FullSearch:
    def __init__(self, batch_size, sylls, dict):
        self.batch_size = batch_size
        self.sylls = sylls
        self.dict = dict
        self.scorevals = np.array((batch_size), dtype='float32')
        self.scorepaths = np.array((batch_size, sylls), dtype='int32')

    def endbatch(self, i, batchpaths, batchvals, lower):
        
        # print('Batch[{}], peak {}: '.format(i, lower))
        last = i % self.batch_size
        if lower == 0.0:
            self.scorevals = batchvals + 0.0
            self.scorepaths = batchpaths + 0
            sortind = np.argsort(self.scorevals)
            self.scorevals = self.scorevals[sortind]
            self.scorepaths = self.scorepaths[sortind]
            return (np.min(self.scorevals), np.max(self.scorevals))    


        sortind = np.argsort(batchvals)
        batchvals = batchvals[sortind]
        batchpaths = batchpaths[sortind]

        # build double-wide arrays
        valset = np.zeros((self.batch_size * 2), dtype='float32')
        valset[:self.batch_size] = self.scorevals[:self.batch_size]
        valset[self.batch_size:] = batchvals[:self.batch_size]
        # sort double-wides, pick best half
        sortind = np.argsort(valset, kind='mergesort')[self.batch_size:self.batch_size*2]
        self.scorevals = valset[sortind]
        pathset = np.zeros((self.batch_size * 2, sylls), dtype='int32')
        pathset[:self.batch_size][:] = self.scorepaths[:self.batch_size][:]
        pathset[self.batch_size:][:] = batchpaths[:self.batch_size][:]
        self.scorepaths = pathset[sortind]
        # re-sort scores
        sortind = np.argsort(self.scorevals)
        self.scorevals = self.scorevals[sortind]
        self.scorepaths = self.scorepaths[sortind]

        return (np.min(self.scorevals), np.max(self.scorevals))

    def mainloop(self, predict):
        vals = np.zeros((self.sylls), dtype='float32')
        last_lower = 0.0
        last_peak = 0.0
        last_better = 0
        i = self.batch_size
        skips = 0
        breakouts = 0
        for x in product(np.arange(dict, dtype='int32'), repeat=sylls):
            if i % self.batch_size == 0:
                batchpaths = np.zeros((self.batch_size, self.sylls), dtype='int32')
                #batchvals = np.zeros((self.batch_size), dtype='float32')
                indices = np.arange(sylls, dtype='int32')
            x = list(x)
            #print('indices: ', x)
            batchpaths[i % self.batch_size] = x
            i += 1
            #print('batchpaths.shape: ', batchpaths.shape)
            #print('batchpaths: ', batchpaths)
            #print('predict.shape: ', predict.shape)
            #print('predict: ', predict)
            batchvals = np.sum(predict[indices, batchpaths], axis=-1)
            newhi = np.max(batchvals)
            if newhi < last_lower:
                skips += 1
                continue
            (lo, hi) = self.endbatch(i - self.batch_size, batchpaths, batchvals, last_lower)
            #if lo > last_lower:
            #    last_lower = lo
            #    last_better = i // self.batch_size
            if hi > last_peak:
                last_peak = hi
                last_better = i // self.batch_size
            last_lower = lo
            last_peak = hi
            if i // self.batch_size - 50 > last_better:
                #print('batch[{}]: break out: lo {}, hi {}'.format(i, lo, hi))
                breakouts += 1
                #break
        return (skips, breakouts)


if __name__ == "__main__":
    sylls = 7
    dict=5
    predict = np.random.random((sylls, dict))
    fb = FullSearch(sylls * sylls * sylls, sylls, dict)
    fb.mainloop(predict)
    print('score[0]: {}'.format(fb.scorevals[0]))
    print('paths[0]: {}'.format(fb.scorepaths[0]))
    print('score[-1]: {}'.format(fb.scorevals[-1]))
    print('paths[-1]: {}'.format(fb.scorepaths[-1]))
    

