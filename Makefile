# Topic Modeling Ind. Study
# Virginia Partridge
# Spring 2021

TXT_CORPUS := ~/workspace/RussianNovels/corpus

corpus:
	ls -d $(TXT_CORPUS)

%.tsv: corpus
	python topic_modeling/preprocessing.py $@ $<