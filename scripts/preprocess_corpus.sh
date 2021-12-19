#!/bin/bash
# Convenience wrapper around stemming pre-processor.

# Corpus name & original format corpus location
#corpus="tiger"
#corpus_source="/home/virginia/workspace/tiger_corpus2.2/tiger_release_aug07.corrected.16012013.xml /home/virginia/workspace/tiger_corpus2.2/TIGER2.2.doc/documents.tsv"

#corpus="rnc"
#corpus_source="/home/virginia/workspace/RNC_million"

corpus="opencorpora"
corpus_source="/home/virginia/workspace/opencorpora/annot.opcorpora.xml"

# The outputs of the corpus_preprocessing.py script
corpus_root="/home/virginia/workspace/topic-modeling-study/${corpus}"

if [ ! -d $corpus_root ]
then
	mkdir -p $corpus_root
fi

python -u -m topic_modeling.corpus_preprocessing --corpus_in $corpus_source --output_dir $corpus_root --corpus_name $corpus > $corpus_root/$corpus.out