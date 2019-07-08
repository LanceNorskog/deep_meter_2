import numpy as np

from itertools import product

''' given predictions.shape=(N, sylls, dict), find the top #pool paths and their scores '''
class FullSearch:
    def __init__(self, pool, sylls, dict):
        self.pool = pool
        self.sylls = sylls
        self.dict = dict

    def endbatch(self, i, batchpaths, batchvals, lower):
        
        # print('Batch[{}], peak {}: '.format(i, lower))
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

        batch_size = len(batchvals)

        # build arrays of both sets of valls, do argsort
        valset = np.zeros((self.pool + batch_size), dtype='float32')
        valset[:self.pool] = self.scorevals[:self.pool]
        valset[self.pool:] = batchvals[:batch_size]
        sortind = np.flip(np.argsort(valset, kind='mergesort'))[:self.pool]
        self.scorevals = valset[sortind]

        # save matching paths
        pathset = np.zeros((self.pool + batch_size, self.sylls), dtype='int32')
        pathset[:self.pool][:] = self.scorepaths[:self.pool][:]
        pathset[self.pool:][:] = batchpaths[:batch_size][:]
        self.scorepaths = pathset[sortind]
        # re-sort scores
        #sortind = np.argsort(self.scorevals)
        #self.scorevals = self.scorevals[sortind]
        #self.scorepaths = self.scorepaths[sortind]

        return (np.min(self.scorevals), np.max(self.scorevals))

    def mainloop(self, predict):
        self.scorevals = np.array((self.pool), dtype='float32')
        self.scorepaths = np.array((self.pool, self.sylls), dtype='int32')
        vals = np.zeros((self.sylls), dtype='float32')
        last_lower = 0.0
        last_peak = 0.0
        last_better = 0
        i = self.pool
        skips = 0
        breakouts = 0
        for x in product(np.arange(self.dict, dtype='int32'), repeat=self.sylls):
            if i % self.pool == 0:
                batchpaths = np.zeros((self.pool, self.sylls), dtype='int32')
                #batchvals = np.zeros((self.pool), dtype='float32')
                indices = np.arange(self.sylls, dtype='int32')
            x = list(x)
            #print('indices: ', x)
            batchpaths[i % self.pool] = x
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
            (lo, hi) = self.endbatch(i - self.pool, batchpaths, batchvals, last_lower)
            #if lo > last_lower:
            #    last_lower = lo
            #    last_better = i // self.pool
            if hi > last_peak:
                last_peak = hi
                last_better = i // self.pool
            last_lower = lo
            last_peak = hi
            #if i // self.pool - 50 > last_better:
                #print('batch[{}]: break out: lo {}, hi {}'.format(i, lo, hi))
                #breakouts += 1
                #break
        return (skips, breakouts)


if __name__ == "__main__":
    _sylls = 7
    _dict=5
    predict = np.random.random((_sylls, _dict))
    fb = FullSearch(20, _sylls, _dict)
    fb.mainloop(predict)
    print('score[0]: {}'.format(fb.scorevals[0]))
    print('paths[0]: {}'.format(fb.scorepaths[0]))
    print('score[-1]: {}'.format(fb.scorevals[-1]))
    print('paths[-1]: {}'.format(fb.scorepaths[-1]))
    print('min {}, max {}'.format(np.min(fb.scorevals), np.max(fb.scorevals)))
    

