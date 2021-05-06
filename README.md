# topic-modeling-study
Independent study on topic modeling with a focus on investigating non-English literary corpora.

# Dependencies for experiments
- [Mallet](http://mallet.cs.umass.edu) should be installed and available in the path
- [Make](https://en.wikipedia.org/wiki/Make_(software)) to easily build corpus and topic experiments (tested with GNU Make)

# Installation for python library
TODO: Nice setup.py configuration.
In the mean time:
1. Create and activate python3 virtual environment. With conda is `conda create -n topic_modeling python=3.7`, then `conda activate topic_modeling` for an environment named `topic_modeling`
2. Install specific requirements: `pip install -r requirements.txt`
3. Run `pip install .` using the python3 within the virtual environment of your choice.

# Build tools
## Building the corpus
Running `make corpus` will produce the following outputs:
- A `.tsv` format of the corpus that is easily fed to Mallet with the columns: doc_id, label (author), text
- A `.mallet` file for the full corpus without pruning and a `counts.tsv` file listing vocab with term and document counts
- A `pruned.mallet` file for the full corpus with the pruning settings determined by `MIN_TERM_FREQ` (passed to Mallet's `--prune-count`), `MIN_IDF` (passed to Mallet's `--min-idf`), `TOKEN_PATTER` (passed to Mallet's `--token-regex`) and a `counts.tsv` file listing vocab with term and document counts for the pruned corpus.

Required options:
- `TXT_CORPUS`: a folder with raw UTF-8 text files.
- `CORPUS_TARGET`: the target folder where all outputs will be written to

## Builds to generate basic corpus analytics
TODO: counts.tsv, vocab.txt, stopped.txt and author correlation studies


## Stemming the corpus
The stemming/lemmatization options from `topic_modeling.stemming.py` are: `pymorphy2`, `pymystem3`, `snowball`, `stanza`, `truncate`
For make, you should specify one of these in `STEM_METHOD`

Running `make stemmed_corpus` will produce a folder named with the corpus target and stemming method containing:
- A `.tsv` file in the Mallet format containing the same documents as the `.tsv` that was produced by `make corpus`, but with stemmed tokens
- A `.tsv` that tracks the counts of original token and stem/lemma pairs broken down by author. This will make it easier to compare the effects of stemming on unique token and token length.
- The various `.mallet` and `counts.tsv` term and document frequency files produced for the stemmed tokens. Uses the same Mallet min-idf, tokenization, etc... settings for import and pruning (see the `make corpus` description above).

## Building experiments
Running `make experiment` will run Mallet's `train-topics` command with the pruned corpus as input. Running `make stemmed_corpus_experiment` will run the `train-topics` command with the pruned, stemmed corpus as input.
Running `make stemmed_post_proc_experiment` will do stemming as post-processing on the Mallet state file, then run `train-topics` with the `--no-inference`, finally generating metrics from that.

All Mallet output will be put in a subfolder of the corpus target or stemmed corpus folder and will be named with the number of topics and iterations.

Required options:
- `NUM_TOPICS`: passed to Mallet's `--num-topics`
- `NUM_ITERS`: passed to Mallet's `--num-iterations`
- `OPTIMIZE_INTERVAL`: passed to Mallet's `--optimize-interval`
- `OPTIMIZE_BURN_IN`: passed to Mallet's `--optimize-burn-in`

## Clean
Running `make clean` removes the entire `CORPUS_TARGET` and stemmed output folders and all the experiments and corpus files it contains. Using `make clean_experiments` will remove all experiments folders.