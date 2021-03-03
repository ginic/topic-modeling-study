# Topic Modeling Ind. Study
# Virginia Partridge
# Spring 2021
SHELL := bash

# Source of raw text files
TXT_CORPUS := ~/workspace/RussianNovels/corpus
#TXT_CORPUS := libru_russian/original

# Dir name for corpus & Mallet output
CORPUS_TARGET := russian_novels
#CORPUS_TARGET := libru_russian

# Path to Authorless TMs repo
# TODO: Target to output correlations files
AUTHORLESS_TMS := ~/workspace/authorless-tms

# Mallet corpus feature settings for info & pruning
# Mallet term frequency and doc freqency pruning settings
MIN_TERM_FREQ := 5
# For Mallet, a given term's idf = ln(|corpus|/doc_freq), so 1.39 is 25% of corpus
MIN_IDF := 1.39
FEATURE_SUFFIX := counts.tsv

# Topic modeling experiments with default settings
MALLET_IMPORT_FLAGS := --keep-sequence
NUM_TOPICS := 100
NUM_ITERS := 1000
OPTIMIZE_INTERVAL := 20
OPTIMIZE_BURN_IN := 50
MALLET_TOPIC_FLAGS := --num-topics $(NUM_TOPICS) --num-iterations $(NUM_ITERS) --optimize-interval $(OPTIMIZE_INTERVAL) --optimize-burn-in $(OPTIMIZE_BURN_IN)

# Naming conventions for topic models
TOPIC_EXPERIMENT_ID := $(NUM_TOPICS)topics_$(NUM_ITERS)iters

# Preprocessing UTF-8 text files to Mallet TSV format
%.tsv: $(TXT_CORPUS)
	mkdir -p $(@D)
	python topic_modeling/preprocessing.py $@ $<
	@echo "Number of original files:"
	@echo $(words $(wildcard $</*.txt))
	@echo "Author ids in output:"
	cut -f2 $@ | sort | uniq | wc -l
	@echo "Novel ids in output:"
	cut -f1,2 -d_ $@ | sort | uniq | wc -l


# Import TSV data to Mallet format
%.mallet: %.tsv
	mallet import-file $(MALLET_IMPORT_FLAGS) --input $< --output $@

# Prunes vocabulary of corpus using given min-idf and min term frequency settings
# Also outputs all features with tf and df for each vocab term, sorted by decreasing df
%_pruned.mallet: %.mallet
	mallet prune --input $< --output $@ --min-idf $(MIN_IDF) --prune-count $(MIN_TERM_FREQ)

# Print out corpus term frequency and document frequency stats
%_$(FEATURE_SUFFIX): %.mallet
	mallet info --input $< --print-feature-counts | sort --key=3 --reverse --numeric > $@

# Just authorless-tms/get_vocab.sh: Get vocabulary sorted alphabetically (no counts)
%_vocab.txt: %.mallet
	mallet info --input $< --print-feature-counts | cut -f 1 | sort -k 1 > $@

# Build list of stop words by comparing the pruned and original vocabs
%_stopped.txt: %_vocab.txt %_pruned_vocab.txt
	comm -23 $^ > $@

# Build a topic model and save topic state from the pruned corpus
# These are probably fragile, don't use with parallel make
%_$(TOPIC_EXPERIMENT_ID): %_pruned.mallet
	mkdir -p $@
	$(eval file_base := $(addsuffix /$(notdir $@),$@))
	$(eval state := $(addsuffix .gz,$(file_base)))
	$(eval output_model := $(addsuffix .model,$(file_base)))
	$(eval doc_topics := $(addsuffix _doc_topics.txt,$(file_base)))
	$(eval topic_keys := $(addsuffix _topic_keys.txt,$(file_base)))
	$(eval top_docs := $(addsuffix _top_docs.txt, $(file_base)))
	mallet train-topics $(MALLET_TOPIC_FLAGS) --input $< --output-state $(state) --output-model $(output_model) --output-doc-topics $(doc_topics) --output-topic-keys $(topic_keys) --output-topic-docs $(top_docs)

# Force all topic modeling files to depend on the output state file
%.gz %.model %_doc_topics.txt %_topic_keys.txt : %
	@test ! -f $@ || touch $@
	@test -f $@ || rm -f $<
	@test -f $@ || $(MAKE) $(AM_MAKEFLAGS) $<

# Run an experiment with default corpus and topic model settings
experiment: $(CORPUS_TARGET)/$(CORPUS_TARGET)_$(TOPIC_EXPERIMENT_ID)

# Build both full and pruned Mallet corpora with default corpus settings
# Sorry about this mess -_-
corpus: $(CORPUS_TARGET)/$(CORPUS_TARGET)_pruned.mallet $(CORPUS_TARGET)/$(CORPUS_TARGET)_pruned_$(FEATURE_SUFFIX) $(CORPUS_TARGET)/$(CORPUS_TARGET).mallet $(CORPUS_TARGET)/$(CORPUS_TARGET)_$(FEATURE_SUFFIX) $(CORPUS_TARGET)/$(CORPUS_TARGET)_vocab.txt $(CORPUS_TARGET)/$(CORPUS_TARGET)_pruned_vocab.txt  $(CORPUS_TARGET)/$(CORPUS_TARGET)_stopped.txt

# Cleans up the default corpus target
clean:
	rm -r $(CORPUS_TARGET)

# Cleans up experiment folders only
clean_experiments:
	rm -r $(CORPUS_TARGET)/$(CORPUS_TARGET)_*topics_*iters


.PHONY: clean experiment corpus clean_experiments

# Don't ever clean up .tsv or .mallet files
.PRECIOUS: %.tsv %.mallet %_pruned.mallet %$(FEATURE_SUFFIX)