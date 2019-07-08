

def a():
    yield 1

def b():
    for x in a():
        yield x

for y in b():
    print(y)
