# Topic Modeling Ind. Study
# Virginia Partridge
# Spring 2021
SHELL := bash

# Source of raw text files
TXT_CORPUS := ~/workspace/RussianNovels/corpus

# Dir name for corpus & Mallet output
CORPUS_TARGET := russian_novels

MALLET_IMPORT_FLAGS :=
NUM_TOPICS := 100
NUM_ITERATIONS := 100
MALLET_TOPIC_FLAGS := --num_topics $(NUM_TOPICS) --num_iterations $(NUM_ITERATIONS)
TOPIC_MODEL_PREFIX := topic_model


# Preprocessing UTF-8 text files to Mallet TSV format
%.tsv: $(TXT_CORPUS)
	mkdir -p $(@D)
	python topic_modeling/preprocessing.py $@ $<
	@echo "Number of original files:"
	@echo $(words $(wildcard $</*.txt))
	@echo "File ids in output:"
	cut -f2 $@ | sort | uniq | wc -l

# Import TSV data to Mallet format
%.mallet: %.tsv
	mallet import-file $(MALLET_IMPORT_FLAGS) --input $< --output $@

# Topic modeling experiments with default settings
TOPIC_EXPERIMENT_ID := $(addsuffix _$(NUM_TOPICS)topics,$(TOPIC_MODEL_PREFIX))
TOPIC_EXPERIMENT_ID := $(addsuffix _$(NUM_ITERS)iters,$(TOPIC_EXPERIMENT_ID))

# Build a topic model and save topic state
%_$(TOPIC_EXPERIMENT_ID).gz: %.mallet
	$(eval experiment_id := $(basename $@))
	$(eval output_model := $(addsuffix .model,$(experiment_id))
	$(eval doc_topics := $(addsuffix _doc_topics.txt,$(experiment_id))
	$(eval topic_keys := $(addsuffix _topic_keys.txt,$(experiment_id))
	mallet train-topics $(MALLET_TOPIC_FLAGS) --input $< --output-state $@ --output-model $(output_model)--output-doc_topics $(doc_topics) --output-topic-keys $(topic_keys)

# Force all topic modeling files to depend on the output state file
 %.model %_doc_topics.txt %_topic_keys.txt : %.gz
	@test ! -f $@ || touch $@
	@test -f $@ || rm -f $<
	@test -f $@ || $(MAKE) $(AM_MAKEFLAGS) $<

# Run an experiment with default corpus and topic model settings
default_experiment: $(CORPUS_TARGET)/$(CORPUS_TARGET)_$(TOPIC_EXPERIMENT_ID).gz

# Build a Mallet corpus with default corpus settings
default_corpus: $(CORPUS_TARGET)/$(CORPUS_TARGET).mallet

# Cleans up the default corpus target
clean:
	rm -r $(CORPUS_TARGET)

.PHONY: clean default_experiment default_corpus