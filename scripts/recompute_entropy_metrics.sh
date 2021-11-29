#!/bin/bash
# Just get entropy metrics over a set of experiments

# Mallet topic model options remaining constant
NUM_ITERS=1000
OPTIMIZE_INTERVAL=20
OPTIMIZE_BURN_IN=50

# Corpus name
corpus="rnc"

# The outputs of the corpus_preprocessing.py script
corpus_root="/home/virginia/workspace/topic-modeling-study/${corpus}"
oracle_gz="${corpus_root}/${corpus}_oracle/${corpus}_oracleAnalysis.gz"

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


# Number of topics
for t in {50..100..50}
do
    # Build 10 experiments for each stemmer
    for stemmer in ${stemmers[@]}
    do
        echo "Starting $stemmer treatment"
        stem_prefix=${corpus}_${stemmer}
        for i in {0..9}
        do
            # Define Mallet outputs
            echo "Training stemmer $stemmer model $i with $t topics"
            stem_prefix=${corpus}_${stemmer}
            model_dir_out="${corpus_root}/${stem_prefix}/${stem_prefix}_${t}_topics_${NUM_ITERS}_iters_${i}"
            state_file=$model_dir_out/${stem_prefix}_state.gz

            echo "Calculating entropy metrics"
            python topic_modeling/mallet_parser.py slot-entropy $state_file $oracle_gz $model_dir_out $stem_prefix
        done
    done
done