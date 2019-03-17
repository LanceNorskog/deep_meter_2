# language models

from SyllableTree import SyllableTree
import pickle


# No influence from language model
class NullModel:
  def __init__(self):
    pass

  def getUnigramProb(self, syll):
    return 1.0

  def getBigramProb(self, s1, s2):
    return 1.0

  def getNextSylls(self, syll):
    return []


# Syllable unigrams & bigram probabilities. Can support word-level, all sylls, or unigram-only depending on how built.
# Uses syllable encodings, not syllable strings, for speed in use by Beam Search
class SyllableModel:

    def __init__(self):
        self.numSylls=0
        self.numUniqueSylls=0
        self.smoothing=True
        self.addK=1.0 if self.smoothing else 0.0
        self.unigrams={}
        self.bigrams={}
        self.tree=SyllableTree() 

    # 'the' [ 1 ]
    # 'themselves' [ 43, 128 ]
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

    def getNextSylls(self, syll_list):
        "syll_list is 1,2 of syllable encodings, returned by PrefixTree"
        return self.tree.getNextSyllables(syll_list)

    def isSyll(self, syll):
        return syll in self.unigrams
        
    def isWord(self, sylls):
        return self.tree.isWord(sylls)
        
    def getUnigramProb(self, syll):
        "prob of seeing syll."
        val=self.unigrams.get(syll)
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


    def saveModel(self, file="blobs/wordmodel.pkl"):
        data = [self.unigrams, self.bigrams, self.tree, self.numUniqueSylls]
        with open(file, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def loadModel(self, file="blobs/wordmodel.pkl"):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        self.unigrams = data[0]
        self.bigrams = data[1]
        self.tree = data[2]
        self.numUniqueSylls = data[3]

if __name__=='__main__':
    sm=SyllableModel()
    sm.addWord('the', [1]) 
    sm.addWord('the', [2])
    sm.addWord('them', [3])
    sm.addWord('themselves', [3, 4]) 
    sm.finishSentences()
    sm.tree.dump()
    sm.saveModel()
    sm2 = SyllableModel()
    sm2.loadModel()
    sm2.tree.dump()
    print('getNextSylls:', sm2.getNextSylls([0]))
    print('getNextSylls:', sm2.getNextSylls([3]))
    print('isSyll:', sm2.isSyll(1))
    print('getBigramProb:', sm2.getBigramProb(3, 4))
    
