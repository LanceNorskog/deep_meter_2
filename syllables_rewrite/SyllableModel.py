from __future__ import print_function
from __future__ import division
import re
from SyllableTree import SyllableTree

def flatten(data):
  return list(chain.from_iterable(data))

class SyllableModel:
    "unigram/bigram LM, add-k smoothing"
    def __init__(self):
        self.sylls=[]
        self.numSylls=0
        self.numUniqueSylls=0
        self.smoothing=True
        self.addK=1.0 if self.smoothing else 0.0
        self.unigrams={}
        self.bigrams={}
        self.tree=SyllableTree() 
        

    # 'the', [ 'DH AH', ...]
    def addWord(self, word, sylls):
        self.numSylls += len(sylls)
        # create unigrams
        for syll in sylls:
            if syll not in self.unigrams:
                self.unigrams[syll]=0
                self.unigrams[syll] += 1
        
        # create unnormalized bigrams
        for i in range(len(sylls)-1):
            s1=sylls[i]
            s2=sylls[i+1]
            if s1 not in self.bigrams:
                self.bigrams[s1]={}
            if s2 not in self.bigrams[s1]:
                self.bigrams[s1][s2]=self.addK # add-K
                self.bigrams[s1][s2]+=1

        self.tree.addWord(word, sylls)
        

    def finishSentences(self):
        self.numUniqueSylls=len(set(self.unigrams))
        for syll in self.unigrams.keys():
            self.unigrams[syll]/=self.numSylls

        #normalize bigrams 
        for s1 in self.bigrams.keys():
            # sum up
            probSum=self.numUniqueSylls*self.addK # add-K smoothing
            for s2 in self.bigrams[s1].keys():
                probSum+=self.bigrams[s1][s2]
            # and divide
            for s2 in self.bigrams[s1].keys():
                self.bigrams[s1][s2] /= probSum
        
    def getNextSylls(self, sylls):
        "text must be prefix of a syll"
        return self.tree.getNextSyllables(sylls)

    def isSyll(self, syll):
        return syll in self.unigrams
        
    def isWord(self, sylls):
        return self.tree.isWord(sylls)
        
    def getUnigramProb(self, w):
        "prob of seeing syll w."
        w=w.lower()
        val=self.unigrams.get(w)
        if val!=None:
            return val
        return 0
    
    def getBigramProb(self, s1, s2):
        "prob of seeing sylls s1 s2 next to each other."
        val1=self.bigrams.get(s1)
        if val1!=None:
            val2=val1.get(s2)
            if val2!=None:
                return val2
            return self.addK/(self.getUnigramProb(s1)*self.numUniqueSylls+self.numUniqueSylls)
        return 0


if __name__=='__main__':
    sm=SyllableModel()
    sm.addWord('the', ['DH AH']) 
    sm.addWord('the', ['DH AE'])
    sm.addWord('them', ['DH EH M'])
    sm.addWord('themselves', ['DH EH M', 'S EH L VZ']) 
    sm.finishSentences()
    sm.tree.dump()
    print('getNextSylls:', sm.getNextSylls(['BLUH']))
    print('getNextSylls:', sm.getNextSylls(['DH EH M']))
    print('isSyll:', sm.isSyll('DH AH'))
    print('getBigramProb:', sm.getBigramProb('DH EH M', 'S EH L VZ'))
    
