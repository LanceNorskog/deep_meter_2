cut -f1 < ../data/gutenberg-poetry-v001.csv | head -10000000 | python classify.py
sort < word_not_found | uniq > word_not_found.txt
rm word_not_found
