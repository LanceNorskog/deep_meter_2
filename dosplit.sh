F=prepped_data/gutenberg.iambic_pentameter.gz 
N=prepped_data/gutenberg.iambic_pentameter
gunzip < $F | head -20000000 | time python splitdata.py 
mv data.train $N.train
mv data.dev $N.dev
mv data.test $N.test
