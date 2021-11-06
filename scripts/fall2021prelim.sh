#!/bin/bash
# Simple script to get train topic models and metric results for a corpus. This assumes you already have converted the corpus to the relevant tsv files using the topic_modeling.corpus_preprocessing script

# Mallet options remain constant
NUM_ITERS=1000
OPTIMIZE_INTERVAL=20
OPTIMIZE_BURN_IN=50

# Corpus name
corpus="tiger"

# Where is the output of the corpus_preprocessing.py script?
corpus_root="/home/virginia/workspace/topic-modeling-study/${corpus}"
oracle_gz="${corpus_root}/${corpus}_oracle/${corpus}_oracleAnalysis.gz"

# Path to the VariationOfInformation.java from https://github.com/xandaschofield/stemmers
voi_java_prog="/home/virginia/workspace/stemmers/VariationOfInformation"

# Russian stemmers
#stemmers=(raw oracle pymystem3 snowball stanza truncate5 truncate6)
# German stemmers
stemmers=(raw oracle spacy snowball stanza truncate5 truncate6)
num_stemmers=${#stemmers[@]}

for stemmer in "${stemmers[@]}"
do
    echo "Building corpus for stemmer: $stemmer"
    stem_prefix=${corpus}_${stemmer}
    input_tsv="${corpus_root}/${stem_prefix}/${stem_prefix}.tsv"
    corpus_out="${corpus_root}/${stem_prefix}/${stem_prefix}.mallet"

    # Topic modeling experiments default settings
    token_pattern="[^\p{Z}]+"
    mallet_import_flags="--keep-sequence --token-regex $token_pattern"

    mallet import-file $mallet_import_flags --input $input_tsv --output $corpus_out
    echo "Mallet corpus created: $corpus_out"
done

# Number of topics
for t in {50..100..50}
do
    voi_out="${corpus_root}/voi_${t}topics"
    mkdir $voi_out
    echo "VOIs will be output to $voi_out"
    # Build 10 experiments for each stemmer
    for stemmer in ${stemmers[@]}
    do
        echo "Training models for $stemmer treatment"
        for i in {0..9}
        do
            # Define Mallet outputs
            echo "Training stemmer $stemmer model $i with $t topics"
            stem_prefix=${corpus}_${stemmer}
            mallet_corpus=${corpus_root}/${stem_prefix}/${stem_prefix}.mallet
            model_dir_out="${corpus_root}/${stem_prefix}/${stem_prefix}_${t}topics_${NUM_ITERS}iters_${i}"
            state_file=$model_dir_out/${stem_prefix}_state.gz
            diag_xml=$model_dir_out/${stem_prefix}_diagnostics.xml
            diag_tsv=$model_dir_out/${stem_prefix}_diagnostics.tsv
            model=$model_dir_out/${stem_prefix}.model
            doc_topics=$model_dir_out/${stem_prefix}_doc_topics.txt
            topic_keys=$model_dir_out/${stem_prefix}_topic_keys.txt
            top_docs=$model_dir_out/${stem_prefix}_top_docs.txt

            mallet_topic_flags="--num-topics $t --num-iterations $NUM_ITERS --optimize-interval $OPTIMIZE_INTERVAL --optimize-burn-in $OPTIMIZE_BURN_IN"

            # Train Mallet topic model
            mkdir -p $model_dir_out
            mallet train-topics $mallet_topic_flags --input $mallet_corpus --output-state $state_file --output-model $model --output-doc-topics $doc_topics --output-topic-keys $topic_keys --output-topic-docs $top_docs --diagnostics-file $diag_xml
            python topic_modeling/mallet_parser.py diagnostics $diag_xml $diag_tsv

            echo "Model trained, results in $diag_tsv"

            echo "Calculating entropy metrics"
            python topic_modeling/mallet_parser.py slot-entropy $state_file $oracle_gz $model_dir_out $stem_prefix
        done

        # Compute VOIS within the treatment
        for j in {0..8}
        do
            for k in {1..9}
            do
                # Which topic models are you comparing?
                modela="${corpus_root}/${stem_prefix}/${stem_prefix}_${t}topics_${NUM_ITERS}iters_${j}/${stem_prefix}_state.gz"
                modelb="${corpus_root}/${stem_prefix}/${stem_prefix}_${t}topics_${NUM_ITERS}iters_${k}/${stem_prefix}_state.gz"
                java $voi_java_prog $t $modela $modelb >> ${voi_out}/${stemmer}_${j}_${stemmer}_${k}.tsv
            done
        done
    done

    # Compute VOIS between treatments
    for (( s1=0; s1<$num_stemmers-1; s1++ ))
    do
        for (( s2=0; s2<$num_stemmers; s2++ ))
        do
            for i in {0..9}
            do
                for j in {0..9}
                do
                    stemmera=${corpus}_${stemmers[$s1]}
                    stemmerb=${corpus}_${stemmers[$s2]}
                    modela="${corpus_root}/${stemmera}/${stemmera}_${t}topics_${NUM_ITERS}iters_${i}/${stemmera}_state.gz"
                    modelb="${corpus_root}/${stemmerb}/${stemmerb}_${t}topics_${NUM_ITERS}iters_${j}/${stemmerb}_state.gz"
                    java $voi_java_prog $t $modela $modelb >> ${voi_out}/${stemmera}_${i}_${stemmerb}_${j}.tsv
                done
            done
        done
    done
done
