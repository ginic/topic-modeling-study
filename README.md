# topic-modeling-study
Independent study on topic modeling with a focus on investigating non-English literary corpora.

# Dependencies for experiments
- [Mallet](http://mallet.cs.umass.edu) should be installed with the class path
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
- A `pruned.mallet` file for the full corpus with the pruning settings determined by `MIN_TERM_FREQ` (passed to Mallet's `--prune-count`) and `MIN_IDF` (passed to Mallet's `--min-idf`) and a `counts.tsv` file listing vocab with term and document counts for the pruned corpus.

Required options:
- `TXT_CORPUS`: a folder with raw UTF-8 text files.
- `CORPUS_TARGET`: the target folder where all outputs will be written to

## Builds to generate basic corpus analytics
TODO: counts.tsv, vocab.txt, stopped.txt and author correlation studies

## Building experiments
Running `make experiment` will run Mallet's `train_topics` command with the pruned corpus as input.
All Mallet output will be put in a subfolder of `CORPUS_TARGET` named with the number of topics and iterations.

Required options:
- `NUM_TOPICS`: passed to Mallet's `--num-topics`
- `NUM_ITERS`: passed to Mallet's `--num-iterations`
- `OPTIMIZE_INTERVAL`: passed to Mallet's `--optimize-interval`
- `OPTIMIZE_BURN_IN`: passed to Mallet's `--optimize-burn-in`

## Clean
Running `make clean` removes the entire `CORPUS_TARGET` folder and all the experiments and corpus files it contains. Using `make clean_experiments` will remove all experiments folders.