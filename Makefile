# Topic Modeling Ind. Study
# Virginia Partridge
# Spring 2021
SHELL := bash

# Source of raw text files
TXT_CORPUS := ~/workspace/RussianNovels/corpus

# Dir name for corpus & Mallet output
CORPUS_TARGET := russian_novels

# Path to Authorless TMs repo
AUTHORLESS_TMS := ~/workspace/authorless-tms

# Topic modeling experiments with default settings
MALLET_IMPORT_FLAGS := --keep-sequence
NUM_TOPICS := 100
NUM_ITERS := 100
MALLET_TOPIC_FLAGS := --num-topics $(NUM_TOPICS) --num-iterations $(NUM_ITERS)

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

# Just authorless-tms/get_vocab.sh
%_vocab.txt: %.mallet
	mallet info --input $< --print-feature-counts | cut -f 1 | sort -k 1 > $@

# Build a topic model and save topic state
# These are probably fragile, don't use with parallel make
%_$(TOPIC_EXPERIMENT_ID): %.mallet
	mkdir -p $@
	$(eval file_base := $(addsuffix /$(notdir $@),$@))
	$(eval state := $(addsuffix .gz,$(file_base)))
	$(eval output_model := $(addsuffix .model,$(file_base)))
	$(eval doc_topics := $(addsuffix _doc_topics.txt,$(file_base)))
	$(eval topic_keys := $(addsuffix _topic_keys.txt,$(file_base)))
	mallet train-topics $(MALLET_TOPIC_FLAGS) --input $< --output-state $(state) --output-model $(output_model) --output-doc-topics $(doc_topics) --output-topic-keys $(topic_keys)

# Force all topic modeling files to depend on the output state file
%.gz %.model %_doc_topics.txt %_topic_keys.txt : %
	@test ! -f $@ || touch $@
	@test -f $@ || rm -f $<
	@test -f $@ || $(MAKE) $(AM_MAKEFLAGS) $<

# Run an experiment with default corpus and topic model settings
default_experiment: $(CORPUS_TARGET)/$(CORPUS_TARGET)_$(TOPIC_EXPERIMENT_ID)

# Build a Mallet corpus with default corpus settings
default_corpus: $(CORPUS_TARGET)/$(CORPUS_TARGET).mallet

# Cleans up the default corpus target
clean:
	rm -r $(CORPUS_TARGET)

.PHONY: clean default_experiment default_corpus

# Don't ever clean up .tsv or .mallet files
.PRECIOUS: %.tsv %.mallet