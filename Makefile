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

# Fill in choice of stemmer here: 'pymorphy2', 'pymystem3', 'snowball', 'stanza', 'truncate'
STEM_METHOD := stanza
STEM_CORPUS := $(CORPUS_TARGET)_$(STEM_METHOD)

# Path to Authorless TMs repo
# TODO: Target to output correlations files
AUTHORLESS_TMS := ~/workspace/authorless-tms

# Mallet corpus feature settings for info & pruning
# Mallet term frequency and doc freqency pruning settings
MIN_TERM_FREQ := 5
# For Mallet, a given term's idf = ln(|corpus|/doc_freq) & Mallet takes care of
# the negative, so 1.39 removes words in more than 25% of documents
# If any questions on maxIDF/maxIDF, check https://github.com/ginic/Mallet/blob/master/src/cc/mallet/classify/tui/Vectors2Vectors.java, lines 142-148
MIN_IDF := 1.39
FEATURE_SUFFIX := counts.tsv
TOKEN_PATTERN := "\p{L}+[\p{P}\p{L}]+\p{L}|\p{L}+"
# Topic modeling experiments with default settings
MALLET_IMPORT_FLAGS := --keep-sequence --token-regex $(TOKEN_PATTERN)
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
	python topic_modeling/preprocessing.py $@ $< --pickle-counts $*_tokens.pickle > $*_preprocessing.out
	tail -n 4 $*_preprocessing.out
	@echo "Number of original files:"
	@echo $(words $(wildcard $</*.txt))
	@echo "Author ids in output:"
	cut -f2 $@ | sort | uniq | wc -l
	@echo "Novel ids in output:"
	cut -f1,2 -d_ $@ | sort | uniq | wc -l
	echo "Total docs:" >> $*_corpus_stats.txt
	wc -l $@ >> $*_corpus_stats.txt


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
	echo "Total tokens $@:" >> $*_corpus_stats.txt
	cut -f2 $@ | awk '{s+=$$0} END {print s}' >> $*_corpus_stats.txt

# Just authorless-tms/get_vocab.sh: Get vocabulary sorted alphabetically (no counts)
%_vocab.txt: %.mallet
	mallet info --input $< --print-feature-counts | cut -f 1 | sort -k 1 > $@
	echo "Vocab size $@:" >> $*_corpus_stats.txt
	wc -l $@ >> $*_corpus_stats.txt

# Build list of stop words by comparing the pruned and original vocabs
%_stopped.txt: %_vocab.txt %_pruned_vocab.txt
	comm -23 $^ > $@

# Build a stemmed version of the Mallet corpus
$(STEM_CORPUS)/$(STEM_CORPUS).tsv: $(CORPUS_TARGET)/$(CORPUS_TARGET).tsv
	mkdir -p $(STEM_CORPUS)
	python topic_modeling/stemming.py $< $@ $(STEM_CORPUS)/$(STEM_CORPUS)_lemma_counts.tsv --lemmatizer $(STEM_METHOD) > $(STEM_CORPUS)/stemming.out
	echo "Total stemmed docs:" >> $(STEM_CORPUS)/$(STEM_CORPUS)_corpus_stats.txt
	wc -l $@ >> $(STEM_CORPUS)/$(STEM_CORPUS)_corpus_stats.txt


# Build a topic model and save topic state from the pruned corpus
# Also produces authorless topic models output for the experiment
# These are probably fragile, don't use with parallel make
%_$(TOPIC_EXPERIMENT_ID): %_pruned.mallet %_pruned_vocab.txt %.tsv
	mkdir -p $@
	$(eval file_base := $(addsuffix /$(notdir $@),$@))
	$(eval state := $(addsuffix .gz,$(file_base)))
	$(eval output_model := $(addsuffix .model,$(file_base)))
	$(eval doc_topics := $(addsuffix _doc_topics.txt,$(file_base)))
	$(eval topic_keys := $(addsuffix _topic_keys.txt,$(file_base)))
	$(eval top_docs := $(addsuffix _top_docs.txt, $(file_base)))
	$(eval diagnostics := $(addsuffix _diagnostics.xml,$(file_base)))
	$(eval author_corrs := $(addsuffix _author_correlation.tsv, $(file_base)))
	mallet train-topics $(MALLET_TOPIC_FLAGS) --input $< --output-state $(state) --output-model $(output_model) --output-doc-topics $(doc_topics) --output-topic-keys $(topic_keys) --output-topic-docs $(top_docs) --diagnostics-file $(diagnostics)
	python $(AUTHORLESS_TMS)/topic_author_correlation.py --input $*.tsv --vocab $*_pruned_vocab.txt --input-state $(state) --output $(author_corrs)

# Force all topic modeling files to depend on the output state file
%.gz %.model %_doc_topics.txt %_topic_keys.txt %_author_correlation.tsv: %
	@test ! -f $@ || touch $@
	@test -f $@ || rm -f $<
	@test -f $@ || $(MAKE) $(AM_MAKEFLAGS) $<

# Run an experiment with default corpus and topic model settings
experiment: $(CORPUS_TARGET)/$(CORPUS_TARGET)_$(TOPIC_EXPERIMENT_ID)

# Build both full and pruned Mallet corpora with default corpus settings
# Sorry about this mess -_-
corpus: $(CORPUS_TARGET)/$(CORPUS_TARGET)_pruned.mallet $(CORPUS_TARGET)/$(CORPUS_TARGET)_pruned_$(FEATURE_SUFFIX) $(CORPUS_TARGET)/$(CORPUS_TARGET).mallet $(CORPUS_TARGET)/$(CORPUS_TARGET)_$(FEATURE_SUFFIX) $(CORPUS_TARGET)/$(CORPUS_TARGET)_vocab.txt $(CORPUS_TARGET)/$(CORPUS_TARGET)_pruned_vocab.txt  $(CORPUS_TARGET)/$(CORPUS_TARGET)_stopped.txt

# Convenience function for building both the stemmed corpus and mallet files
# for stemmed corpus
stemmed_corpus: $(STEM_CORPUS)/$(STEM_CORPUS).tsv $(STEM_CORPUS)/$(STEM_CORPUS)_pruned.mallet $(STEM_CORPUS)/$(STEM_CORPUS)_pruned_$(FEATURE_SUFFIX) $(STEM_CORPUS)/$(STEM_CORPUS).mallet $(STEM_CORPUS)/$(STEM_CORPUS)_$(FEATURE_SUFFIX) $(STEM_CORPUS)/$(STEM_CORPUS)_vocab.txt $(STEM_CORPUS)/$(STEM_CORPUS)_pruned_vocab.txt  $(STEM_CORPUS)/$(STEM_CORPUS)_stopped.txt

# Builds topic models from stemmed corpus
stemmed_experiment: $(STEM_CORPUS)/$(STEM_CORPUS)_$(TOPIC_EXPERIMENT_ID)

# Cleans up the default corpus target and stemmed corpus target
clean:
	rm -r $(CORPUS_TARGET)
	rm -r $(STEM_CORPUS)

# Cleans up experiment folders only
clean_experiments:
	rm -r $(CORPUS_TARGET)/$(CORPUS_TARGET)_*topics_*iters
	rm -r $(STEM_CORPUS)/$(STEM_CORPUS)_*topics_*iters


.PHONY: clean experiment corpus clean_experiments stemmed_corpus stemmed_experiment

# Don't ever clean up .tsv or .mallet files
.PRECIOUS: %.tsv %.mallet %_pruned.mallet %$(FEATURE_SUFFIX) %_vocab.txt