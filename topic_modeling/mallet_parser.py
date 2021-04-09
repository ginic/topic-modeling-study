"""Functions for dealing with Mallet's diagnostics and state files
"""
import argparse
import xml.etree.ElementTree as ET
import gzip

import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix

import topic_modeling.stemming as stemming

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
            vocab_index[lemma] = len(vocab)
            vocab.append(lemma)
    return vocab, vocab_index


def stem_state_file(in_state_file, out_state_file, vocab_index, stemmer):
    """Creates a new Mallet .gzip state file by post processing vocab elements with stemming.

    :param in_state_file: Mallet .gzip file produced by mallet train-topics --output-state option
    :param out_state_file: Target path for new .gzip file
    :param vocab_index: map of stemmed vocab item to int index
    :param stemmer: a Stemmer object from topic_modeling.stemming
    """
    # TODO Some clean up needed here
    # TODO What do we want to do with topic/term counts?
    n_vocab = len(vocab_index)
    gzip_reader = gzip.open(in_state_file, mode='rt', encoding='utf-8')
    gzip_writer = gzip.open(out_state_file, mode='wt', encoding='utf-8')
    header = gzip_reader.readline()
    gzip_writer.write(header)
    alpha_text = gzip_reader.readline()
    gzip_writer.write(alpha_text)
    alphas = [float(a) for a in alpha_text.strip().split(' : ')[1].split()]
    n_topics = len(alphas)
    beta_text = gzip_reader.readline()
    gzip_writer.write(beta_text)

    topic_term_counts = np.zeros((n_topics, n_vocab))

    current_doc_id = -1
    for i, line in enumerate(gzip_reader):
        fields = line.strip().split()
        doc_id = int(fields[0])
        source = fields[1]
        pos = fields[2]
        term = fields[4]
        topic = int(fields[5])
        if doc_id and doc_id % 1000==0:
            if doc_id != current_doc_id:
                print("Reading terms for doc:", doc_id)
                current_doc_id = doc_id
        stemmed_term = stemmer.single_term_lemma(term)
        stemmed_term_index = vocab_index[stemmed_term]

        output_result = " ".join([str(doc_id), source, pos, str(stemmed_term_index), stemmed_term, str(topic)])
        gzip_writer.write(output_result + "\n")

        topic_term_counts[topic, stemmed_term_index] += 1

    gzip_writer.close()
    gzip_reader.close()
    return topic_term_counts


parser = argparse.ArgumentParser(
    description="Functionality for creating new Mallet state files and reading Mallet diagnostics files."
)
subparsers = parser.add_subparsers(dest='subparser_name')
# TODO better help messages


diagnostics_parser = subparsers.add_parser('diagnostics', help="Parses Mallet diagnostics file to CSV and reports overall statistics")
diagnostics_parser.add_argument('in_xml', help="Mallet diagnostics xml file")
diagnostics_parser.add_argument('out_tsv', help="Target path for diagnostics in TSV format")


state_file_parser = subparsers.add_parser('state-file', help="Work with state files")
state_file_parser.add_argument('in_gz', help="Input gzip file, an existing state file")
state_file_parser.add_argument('input_vocab', help="Text file, vocabulary corresponding to the state file, one term per line.")
state_file_parser.add_argument('out_gz', help="Desired path for new gzip to be created")
state_file_parser.add_argument('--lemmatizer', '-l',
    help='Choice of stemmer/lemmatizer',
    choices=stemming.STEMMER_CHOICES)


if __name__ == "__main__":
    args = parser.parse_args()
    subparser_name = args.subparser_name
    if subparser_name=="state-file":
        stemmer = stemming.pick_lemmatizer(args.lemmatizer)
        stemmed_vocab, stemmed_idx = get_stemmed_vocab(args.input_vocab, stemmer)
        print("Stemmed vocab and index produced. Number of stemmed vocab items:", len(stemmed_vocab))
        # TODO write out new vocab file
        print("Producing stemmed version of", args.in_gz, "to be written to", args.out_gz)
        topic_term_counts = stem_state_file(args.in_gz, args.out_gz, stemmed_idx, stemmer)
    else:
        diagnostics_df = diagnostics_xml_to_dataframe(args.in_xml)
        # Make sure pandas will print everything
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', -1)
        print(diagnostics_df.describe())
        diagnostics_df.to_csv(args.out_tsv, sep="\t")
