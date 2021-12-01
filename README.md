# topic-modeling-study
Independent study on topic modeling with a focus on investigating non-English literary corpora.

# Dependencies for experiments
- python3 and your choice of virtual environment tool
- [Mallet](http://mallet.cs.umass.edu) should be installed and available in the path. It must be the latest version from Github, not the version from the website.
- [Java 8](https://openjdk.java.net/install/), OpenJDK is fine.
- [Variation of Information Java program](https://github.com/ginic/stemmers/blob/master/VariationOfInformation.java) should be downloaded and compiled (using javac). Include the class path to this program in `scripts/build_models_metrics.sh`.
- [wc](https://en.wikipedia.org/wiki/Wc_(Unix)) and [awk](https://en.wikipedia.org/wiki/AWK) for scripts that compute corpus stats.

# Installation for python library
TODO: Nice setup.py configuration.
In the mean time:
1. Create and activate python3 virtual environment. With conda is `conda create -n topic_modeling python=3.7`, then `conda activate topic_modeling` for an environment named `topic_modeling`
2. Install specific requirements: `pip install -r requirements.txt`
3. Run `pip install .` using the python3 within the virtual environment of your choice.

# Build tools
Corpus pre-processing and topic modeling experiments are done using shell scripts in `scripts`.

## Building the corpus
The `topic_modeling/corpus_preprocessing.py` will create Mallet `.tsv` files the raw tokens, oracle lemmas and each appropriate stemmer given the language of the corpus. The `scripts/preprocess_corpus.sh` is a convenience wrapper around the python script.

## Corpus statistics
To generate counts of tokens, word types, type to token ratio and character to token ratio for each stemming treatment, use `scripts/corpus_stats.sh`. It takes the `.tsv` files output from `topic_modeling/corpus_preprocessing.py` and produces a `.tsv`.

## Building experiments
Running `scrips/build_models_metrics.sh` will import data to the Mallet vocab format, then run Mallet's `train-topics` command for 50 and 100 topics. All Mallet output will be put in a subfolder of the corpus target or stemmed corpus folder and will be named with the number of topics and iterations.

Diagnostics from Mallet are also written for each model, as well as the morphological and grammatical entropy metrics generated by `topic_modeing/mallet_parser.py`. This script also produces post-lemmatized (using the oracle as source for lemmas) metrics for each model.

Finally, pairwise variation of information results are computed amongst all the trained topic models.

Required options:
- `NUM_ITERS`: default is 1000, passed to Mallet's `--num-iterations`
- `OPTIMIZE_INTERVAL`: default is 20, passed to Mallet's `--optimize-interval`
- `OPTIMIZE_BURN_IN`: default is 50, passed to Mallet's `--optimize-burn-in`