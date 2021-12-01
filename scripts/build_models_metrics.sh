#!/bin/bash
# Simple script to get train topic models and metric results for a corpus. This assumes you already have converted the corpus to the relevant tsv files using the topic_modeling.corpus_preprocessing script
# Dependencies: Mallet on PATH, VariationOfInformation.java compiled from https://github.com/xandaschofield/stemmers

# Mallet topic model options remaining constant
NUM_ITERS=1000
OPTIMIZE_INTERVAL=20
OPTIMIZE_BURN_IN=50

# Corpus name
corpus="tiger"

# The outputs of the corpus_preprocessing.py script
corpus_root="/home/virginia/workspace/topic-modeling-study/${corpus}"
oracle_gz="${corpus_root}/${corpus}_oracle/${corpus}_oracleAnalysis.gz"

# Path to dir containing compiled VariationOfInformation xandaschofield/stemmers
voi_java_class_path="/home/virginia/workspace/stemmers"

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

# Number of topics - train models and calculate VOIs
for t in {50..100..50}
do
    voi_out="${corpus_root}/voi_${t}_topics"
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
            model_dir_out="${corpus_root}/${stem_prefix}/${stem_prefix}_${t}_topics_${NUM_ITERS}_iters_${i}"
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

            #echo "Calculating entropy metrics"
            python topic_modeling/mallet_parser.py slot-entropy $state_file $oracle_gz $model_dir_out $stem_prefix

            echo "Post-lemmatizing model to produce lemma level metrics"
            # Post lemmatize -> State file -> Seq -> Diagnostics with no inference
            post_lemmatize_prefix="${stem_prefix}_postLemmatized"
            post_lemmmatize_state_file=$model_dir_out/${post_lemmatize_prefix}_state.gz
            post_lemmatize_sequence_file=$model_dir_out/${post_lemmatize_prefix}.mallet
            post_diag_xml=$model_dir_out/${post_lemmatize_prefix}_diagnostics.xml
            post_diag_tsv=$model_dir_out/${post_lemmatize_prefix}_diagnostics.tsv
            python topic_modeling/mallet_parser.py oracle-post-lemmatize $state_file $post_lemmmatize_state_file $oracle_gz
            mallet run cc.mallet.util.StateToInstances --input $post_lemmmatize_state_file --output $post_lemmatize_sequence_file
            mallet train-topics $mallet_topic_flags --input $post_lemmatize_sequence_file --input-state $post_lemmmatize_state_file --no-inference  true --output-model $model_dir_out/${post_lemmatize_prefix}.model --diagnostics-file $post_diag_xml
            python topic_modeling/mallet_parser.py diagnostics $post_diag_xml $post_diag_tsv

        done

        # Compute VOIS within the treatment
        for j in {0..8}
        do
            for (( k=$(( $j+ 1 )); k<10; k++ ))
            do
                # Which topic models are you comparing?
                modela="${corpus_root}/${stem_prefix}/${stem_prefix}_${t}_topics_${NUM_ITERS}_iters_${j}/${stem_prefix}_state.gz"
                modelb="${corpus_root}/${stem_prefix}/${stem_prefix}_${t}_topics_${NUM_ITERS}_iters_${k}/${stem_prefix}_state.gz"
                echo "Computing VOI between $t topics in $modela and $modelb"
                java -cp $voi_java_class_path VariationOfInformation $t $modela $modelb >> ${voi_out}/${stem_prefix}_${j}_${stem_prefix}_${k}.tsv
            done
        done
    done

    # Compute VOIS between treatments
    for (( s1=0; s1<$num_stemmers-1; s1++ ))
    do
        for (( s2=$(( $s1 + 1 )); s2<$num_stemmers; s2++ ))
        do
            for i in {0..9}
            do
                for j in {0..9}
                do
                    stemmera=${corpus}_${stemmers[$s1]}
                    stemmerb=${corpus}_${stemmers[$s2]}
                    modela="${corpus_root}/${stemmera}/${stemmera}_${t}_topics_${NUM_ITERS}_iters_${i}/${stemmera}_state.gz"
                    modelb="${corpus_root}/${stemmerb}/${stemmerb}_${t}_topics_${NUM_ITERS}_iters_${j}/${stemmerb}_state.gz"
                    echo "Computing VOI between $t topics in $modela and $modelb"
                    java -cp $voi_java_class_path VariationOfInformation $t $modela $modelb >> ${voi_out}/${stemmera}_${i}_${stemmerb}_${j}.tsv
                done
            done
        done
    done
done