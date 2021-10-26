#!/bin/bash
# Simple script to get some quick experiment results
corpus="rnc"
stemmers="raw oracle pymystem3 snowball stanza truncate5 truncate6"
for stemmer in $stemmers
do
    INPUT_TSV="/home/virginia/workspace/topic-modeling-study/${corpus}/${corpus}_$stemmer/${corpus}_$stemmer.tsv"
    CORPUS_OUT="/home/virginia/workspace/topic-modeling-study/${corpus}/${corpus}_$stemmer/${corpus}_$stemmer.mallet"

    TOKEN_PATTERN="[^\p{Z}]+"
    # Topic modeling experiments with default settings
    MALLET_IMPORT_FLAGS="--keep-sequence --token-regex $TOKEN_PATTERN"
    NUM_TOPICS=50
    NUM_ITERS=1000
    OPTIMIZE_INTERVAL=20
    OPTIMIZE_BURN_IN=50
    MALLET_TOPIC_FLAGS="--num-topics $NUM_TOPICS --num-iterations $NUM_ITERS --optimize-interval $OPTIMIZE_INTERVAL --optimize-burn-in $OPTIMIZE_BURN_IN"

    mallet import-file $MALLET_IMPORT_FLAGS --input $INPUT_TSV --output $CORPUS_OUT
    echo "Mallet corpus created: $CORPUS_OUT"
    # TODO add for loop for experiments
    MODEL_DIR_OUT="/home/virginia/workspace/topic-modeling-study/${corpus}/${corpus}_$stemmer/${corpus}_${stemmer}_prelim_exp"
    STATE_FILE=$MODEL_DIR_OUT/${corpus}_${stemmer}_state.gz
    DIAG_XML=$MODEL_DIR_OUT/${corpus}_${stemmer}_diagnostics.xml
    DIAG_TSV=$MODEL_DIR_OUT/${corpus}_${stemmer}_diagnostics.tsv
    MODEL=$MODEL_DIR_OUT/${corpus}_$stemmer.model
    DOC_TOPICS=$MODEL_DIR_OUT/${corpus}_${stemmer}_doc_topics.txt
    TOPIC_KEYS=$MODEL_DIR_OUT/${corpus}_${stemmer}_topic_keys.txt
    TOP_DOCS=$MODEL_DIR_OUT/${corpus}_${stemmer}_top_docs.txt


    mkdir -p $MODEL_DIR_OUT
    mallet train-topics $MALLET_TOPIC_FLAGS --input $CORPUS_OUT --output-state $STATE_FILE --output-model $MODEL --output-doc-topics $DOC_TOPICS --output-topic-keys $TOPIC_KEYS --output-topic-docs $TOP_DOCS --diagnostics-file $DIAG_XML
    python topic_modeling/mallet_parser.py diagnostics $DIAG_XML $DIAG_TSV
    echo "Model trained, results in $DIAG_TSV"
done
