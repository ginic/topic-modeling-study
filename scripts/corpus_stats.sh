#!/bin/bash
# Corpus name
corpus="tiger"

# The outputs of the corpus_preprocessing.py script
corpus_root="/home/virginia/workspace/topic-modeling-study/${corpus}"
output_tsv="${corpus_root}/${corpus}_corpus_stats.tsv"
echo "stemmer\ttoken_count\tword_type_count\ttype_to_token_ratio\tchar_to_token_ratio" > $output_tsv

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
    stem_prefix=${corpus}_${stemmer}
    corpus_tsv=${corpus_root}/${stem_prefix}/${stem_prefix}.tsv
    token_count=$(cut -f 3 $corpus_tsv | wc -w)
    word_type=$(cut -f 3 $corpus_tsv | tr ' ' '\n' | sort | uniq | wc -l)
    ttr=$(awk "BEGIN {print $word_type/$token_count}")
    char_count=$(cut -f 3 $corpus_tsv | tr -d ' \n' | sort | uniq | wc -m)
    char_to_token=$(awk "BEGIN {print $char_count/$token_count}")
    echo "$stemmer\t$token_count\t$word_type\t$ttr\t$char_to_token" >> $output_tsv
done
