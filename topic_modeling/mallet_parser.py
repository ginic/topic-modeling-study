"""Functions for dealing with Mallet's diagnostics and state files
.. TODO for working with mallet state and vocab files, start with following authorless tm code and adapt for
stemming distributions
"""

import xml.etree.ElementTree as ET
import gzip

import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix

# Diagnostics XML constants
TOPIC="topic"
TOPIC_ID="id"


def diagnostics_xml_to_dataframe(xml_path):
    """Returns a pandas dataframe with the
    :param xml_path: Path to a Mallet diagnostics file
    """
    model_root = ET.parse(xml_path).getroot()
    topic_metrics = list()
    for t in model_root.findall(TOPIC):
        attributes = {}
        for k,v in t.items():
            if k == TOPIC_ID:
                attributes[k] = int(v)
            else:
                attributes[k] = float(v)
        topic_metrics.append(attributes)

    return pd.DataFrame.from_records(topic_metrics, index=TOPIC_ID)


def get_vocab(vocab_file):
    """Read in the vocabulary from a txt file, one element per line.
    Return list of vocab items and index mapping {term: index in list}.
    Pulled directly from authorless-tms.

    :param vocab_file: file in UTF-8 format, one term per line
    """
    vocab = []
    vocab_index = {}
    for i, line in enumerate(open(vocab_file, mode='r', encoding='utf-8')):
        term = line.strip()
        if term != '':
            vocab.append(term)
            vocab_index[term] = i
    return vocab, vocab_index


def get_stemmed_vocab(vocab_file, stemmer):
    '''Read in the vocabulary from a txt file, one element per line.
    Stems items and maps vocab appropriately to new indices
    Return list of vocab items and index mapping {term: index in list}.

    :param vocab_file: file in UTF-8 format, one term per line
    :param stemmer: a Stemmer object from topic_modeling.stemming
    '''
    vocab = []
    vocab_index = {}
    for i, line in enumerate(open(vocab_file, mode='r', encoding='utf-8')):
        lemma = stemmer.single_term_lemma(line.strip())
        if lemma != '' and lemma not in vocab_index:
            vocab.append(lemma)
            vocab_index[lemma] = i
    return vocab, vocab_index


def stem_state_file(in_state_file, out_state_file, vocab_index, stemmer):
    """Creates a new Mallet .gzip state file by prost processing vocab elements with stemming.

    :param in_state_file: Mallet .gzip file produced by mallet train-topics --output-state option
    :param out_state_file: Target path for new .gzip file
    :param vocab_index: map of vocab item to int index
    :param stemmer: a Stemmer object from topic_modeling.stemming
    """
    n_vocab = len(vocab_index)
    gzip_reader = gzip.open(in_state_file, mode='rt', encoding='utf-8')
    # TODO Write out to new gzip file
    gzip_reader.readline()  # header
    alpha_text = gzip_reader.readline().strip()
    alphas = [float(a) for a in alpha_text.split(' : ')[1].split()]
    n_topics = len(alphas)
    beta_text = gzip_reader.readline().strip()
    beta = float(beta_text.split(' : ')[1])

    # TODO stemming and replacement with new index in new gzip
    topic_term_counts = np.zeros((n_topics, n_vocab))

    current_doc_id = -1
    for i, line in enumerate(gzip_reader):
        fields = line.strip().split()
        doc_id = int(fields[0])
        if doc_id and doc_id % 1000==0:
            if doc_id != current_doc_id:
                print("Reading weights for doc:", doc_id)
                current_doc_id = doc_id
        term = fields[4]
        topic = int(fields[5])

        term_idx = vocab_index[term]

        topic_term_counts[topic, term_idx] += 1

    gzip_reader.close()
    return topic_term_counts


