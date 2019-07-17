# next word given current worD
# boolean matrix of word index vs word index
# in fact, just do array -> set(index) bc of memory size

import numpy as np

class Wordmap:
    def __init__(self, m):
        self.mat = [set()] * m
        for x in range(m):
            self.mat[x] = set()

    def add(self, i, j):
        if self.mat[i] == None:
            self.mat[i] = set()
        print("Adding {} to {}".format(j, self.mat[i]))
        self.mat[i].add(j)

    def get(self, i, j):
        if self.mat[i] == None:
            return True
        else:
            if j in self.mat[i]:
                return True
            else:
                return False

    def print(self):
        print(self.mat)


if __name__ == "__main__":
    wordmap = Wordmap(5)
    print("False: ", wordmap.get(1, 3))
    wordmap.add(1, 3)
    wordmap.add(1, 3)
    print("True: ", wordmap.get(1, 3))
    print("False: ", wordmap.get(2, 3))
    wordmap.print()
