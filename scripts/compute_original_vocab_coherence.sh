#!/bin/bash
# Generates a state file that uses the original untreated (raw) corpus terms for each
# stemmed/lemmatized topic model experiment.
# Compute diagnostics of the model according to the original vocabulary.

corpus="rnc"

# Mallet topic model options remaining constant
NUM_ITERS=1000
OPTIMIZE_INTERVAL=20
OPTIMIZE_BURN_IN=50

# The outputs of the corpus_preprocessing.py script
corpus_root="/home/virginia/workspace/topic-modeling-study/${corpus}"

# This can be any raw model experiment state file, you're just going to grab the terms and vocab indices
raw_gz="${corpus_root}/${corpus}_raw/${corpus}_raw_50_topics_1000_iters_0/${corpus}_raw_state.gz"
raw_mallet="${corpus_root}/${corpus}_raw/${corpus}_raw.mallet"

if [ "$corpus" = "tiger" ]; then
    stemmers=(oracle spacy snowball stanza truncate5 truncate6)
    echo "Using German stemmers: ${stemmers[@]}"
else
    stemmers=(oracle pymystem3 snowball stanza truncate5 truncate6)
    echo "Using Russian stemmers: ${stemmers[@]}"
fi

for t in {50..100..50}
do
	for stemmer in ${stemmers[@]}
	do
		echo "Parsing state for $stemmer treatment"
		for i in {0..9}
		do
			echo "Unstemming $stemmer experiment $i"
			treatment_prefix=${corpus}_${stemmer}
			treatment_model_dir=${corpus_root}/${treatment_prefix}/${treatment_prefix}_${t}_topics_${NUM_ITERS}_iters_${i}
			treatment_state_file=$treatment_model_dir/${treatment_prefix}_state.gz
			unstemmed_prefix=${treatment_prefix}_unstemmed
			unstemmed_state_file=$treatment_model_dir/${unstemmed_prefix}_state.gz
			unstemmed_diagnostics_xml=$treatment_model_dir/${unstemmed_prefix}_diagnostics.xml
			unstemmed_diagnostics_tsv=$treatment_model_dir/${unstemmed_prefix}_diagnostics.tsv
			unstemmed_model=$treatment_model_dir/${unstemmed_prefix}.model

			python topic_modeling/mallet_parser.py unmap-stemmed-state $raw_gz $treatment_state_file $unstemmed_state_file

			mallet_topic_flags="--num-topics $t --num-iterations $NUM_ITERS --optimize-interval $OPTIMIZE_INTERVAL --optimize-burn-in $OPTIMIZE_BURN_IN"
			mallet train-topics $mallet_topic_flags --input $raw_mallet --input-state $unstemmed_state_file --no-inference true --output-model $unstemmed_model --diagnostics-file $unstemmed_diagnostics_xml

			python topic_modeling/mallet_parser.py diagnostics $unstemmed_diagnostics_xml $unstemmed_diagnostics_tsv
		done
	done
done
