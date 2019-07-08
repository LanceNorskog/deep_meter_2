import numpy as np

from itertools import product

''' given predictions.shape=(N, sylls, dict), find the top #pool paths and their scores '''
class FullSearch:
    def __init__(self, pool, sylls, dict):
        if (pool % sylls != 0):
            raise Exception('pool must be a multiple of sylls')
        self.pool = pool
        self.sylls = sylls
        self.dict = dict

    def endbatch(self, i, batchpaths, batchvals, lower):
        #print('check: ', batchpaths)
        
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
        sortind = np.flip(np.argsort(valset))[:self.pool]
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
        i = 0
        skips = 0
        breakouts = 0
        batchpaths = np.zeros((self.pool, self.sylls), dtype='int32')
        #batchvals = np.zeros((self.pool), dtype='float32')
        indices = np.arange(self.sylls, dtype='int32')
        for x in product(np.arange(self.dict, dtype='int32'), repeat=self.sylls):
            x = list(x)
            batchpaths[i % self.pool] = x
            if i % self.pool == 0:
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
            i += 1
        return (skips, breakouts)


if __name__ == "__main__":
    def fib(n):
        out = [0]*n
        i = 2
        out[0] = 1
        out[1] = 1
        for i in range(2, n):
            out[i] = out[i-1] + out[i-2]
        return out

    _sylls = 5
    _dict = 7
    data = fib(_sylls * _dict)
    data[0] = 0
    predict = np.reshape(data, (_sylls, _dict))
    print('predict: ', predict)
    fb = FullSearch(_sylls, _sylls, _dict)
    fb.mainloop(predict)
    print('score[0]: {}'.format(fb.scorevals[0]))
    print('paths[0]: {}'.format(fb.scorepaths[0]))
    print('score[-1]: {}'.format(fb.scorevals[-1]))
    print('paths[-1]: {}'.format(fb.scorepaths[-1]))
    print('min {}, max {}'.format(np.min(fb.scorevals), np.max(fb.scorevals)))
    print('vals {}'.format((fb.scorevals)))
    print('paths {}'.format((fb.scorepaths)))
    

