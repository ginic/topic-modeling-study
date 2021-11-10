#!/bin/bash
# Corpus name
corpus="tiger"

# The outputs of the corpus_preprocessing.py script
corpus_root="/home/virginia/workspace/topic-modeling-study/${corpus}"

if [ "$corpus" = "tiger" ]; then
    # German stemmers
    stemmers=(raw oracle spacy snowball stanza truncate5 truncate6)
    echo "Using German stemmers: ${stemmers[@]}"
else
    # Russian stemmers
    stemmers=(raw oracle pymystem3 snowball stanza truncate5 truncate6)
    echo "Using Russian stemmers: ${stemmers[@]}"
fi

num_stemmers=${#stemmers[@]}

for stemmer in "${stemmers[@]}"
do
# TODO token count, word type count, type to token ration, character to token ratio
done
