from __future__ import print_function
from __future__ import division


class Node:
    "class representing nodes in a prefix tree"
    def __init__(self):
        self.children={} # all child elements beginning with current syllable
        self.word=None
        
    def __str__(self):
        s=''
        for (k,_) in self.children.items():
            s+=k + ','
        return 'word: '+str(self.word)+'; children: '+s

class SyllableTree:
    "syllable->word trie"
    def __init__(self):
        self.root=Node()

    def addWord(self, word, sylls):
        "add arpa list to prefix tree"
        node=self.root
        for i in range(len(sylls)):
            s=sylls[i] # current syllable
            if s not in node.children:
                node.children[s]=Node()
            node=node.children[s]
            isLast=(i+1==len(sylls))
            if isLast:
                node.word = word
                                
    def getNode(self, sylls):
        "get node representing given text"
        node=self.root
        for s in sylls:
            if s in node.children:
                node=node.children[s]
            else:
                return None
        return node

        
    def isWord(self, sylls):
        node=self.getNode(sylls)
        if node:
            return node.word != None
        return False
        
    
    def getNextSyllables(self, sylls):
        "get all syllables which may directly follow given text"
        next=[]
        node=self.getNode(sylls)
        if node:
            for k,_ in node.children.items():
                next.append(k)
        return next
    
    
    def getNextWords(self, sylls):
        "get all words of which given syllable list is a prefix (including the syllable list itself, if it is a word)"
        words=[]
        node=self.getNode(sylls)
        if node:
            nodes=[node]
            prefixes=[sylls]
            while len(nodes)>0:
                # put all children into list
                for k,v in nodes[0].children.items():
                    nodes.append(v)
                    prefixes.append(k)
                
                # is current node a word
                if nodes[0].word != None:
                    words.append(nodes[0].word)
                
                # remove current node
                del nodes[0]
                del prefixes[0]
                
        return words
                
    def dump(self):
        nodes=[self.root]
        while len(nodes)>0:
            # put all children into list
            for _,v in nodes[0].children.items():
                nodes.append(v)
            
            # dump current node
            print(nodes[0])
                
            # remove from list
            del nodes[0]

                
if __name__=='__main__':
    t=SyllableTree() # create tree
    t.addWord('the', ['DH AH']) # add words
    t.addWord('the', ['DH AE']) # add words
    t.addWord('them', ['DH EH M']) # add words
    t.addWord('themselves', ['DH EH M', 'S EH L VS']) # add words
    print(t.getNextSyllables(['DH'])) # chars following 'th'
    print(t.getNextWords(['DH AE'])) # all words of which 'th' is prefix
    print(t.getNextWords(['DH EH M'])) # all words of which 'th' is prefix
    t.dump()
