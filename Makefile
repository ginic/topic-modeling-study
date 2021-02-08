# Topic Modeling Ind. Study
# Virginia Partridge
# Spring 2021
SHELL := bash

TXT_CORPUS := ~/workspace/RussianNovels/corpus

# Preprocessing UTF-8 text files to Mallet TSV format
%.tsv: $(TXT_CORPUS)
	mkdir -p $(@D)
	python topic_modeling/preprocessing.py $@ $<
	@echo "Number of original files"
	$(words $(wildcard $</*.txt))
	@echo "File ids in output"
	cut -f2 $@ | sort | uniq | wc -l

clean:
	@echo  "TODO"


.PHONY: clean