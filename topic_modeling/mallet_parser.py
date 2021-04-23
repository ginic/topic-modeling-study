"""Functions for dealing with Mallet's diagnostics and state files
"""
import argparse
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
import gzip

import numpy as np
import pandas as pd
import pymystem3

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

def write_header_to_morph_analysis_file(filepath, col_name):
    with open(filepath, 'w') as f:
        f.write(f"topic\t{col_name}\tcount\tproportion\n")

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

def entropy(probabilities):
    """Returns the entropy given a list of probabilities for n elements
    """
    return -np.sum(probabilities * np.log2(probabilities))

def append_topic_morphological_analysis_to_file(topic_id, tsv_file, weighted_morph_analysis_counts, total_counts):
    """Writes out morphological analysis of a topic to tsv file in order of descending proportion
    """
    results = []
    for k,v in weighted_morph_analysis_counts.items():
        proportion = v / total_counts
        results.append([topic_id, k, v, proportion])

    sorted_results = sorted(results, key=lambda x: x[3], reverse=True)

    with open(tsv_file, 'a') as f:
        for l in sorted_results:
            f.write("\t".join([str(x) for x in l]) + "\n")


def morphological_entropy_single_topic(mystem, topic_term_counts, topic_id=None, stem_file=None, lemma_file=None, pos_file=None):
    """For a given topic, returns the slot entropy, number of slots, part of speech entropy, number of part of speech tags, lemma entropy, number of lemmas.
    If an outputdir is specified, the pos tags, slots and lemmas will be written

    :param mystem: Pymystem3 instance for determining mrophological analysis
    :param topic_term_counts: {term: count} mapping for a single topic
    :param topic_id: Id of topic, only needed if you're going to write detailed results to a file
    :param stem_file: file to write detailed stem analysis to
    :param lemma_file: file to write detailed lemma analysis to
    :param pos_file: file to write detailed pos analysis to
    """
    weighted_slot_counts = defaultdict(float)
    weighted_lemma_counts = defaultdict(float)
    weighted_pos_counts = defaultdict(float)
    for surface_form, count in topic_term_counts.items():
        # We're just going to grab the first analysis element from pymystem3.
        # This might be wrong in some very small numbers of edge cases where pymystem3
        # tokenization is different (usually involves hyphens).
        analysis = mystem.analyze(surface_form)[0]
        for morph_analysis in analysis['analysis']:
            slot = morph_analysis['gr']
            pos = slot.split("=")[0]
            weight = morph_analysis['wt']
            lemma = morph_analysis['lex']
            if weight != 0:
                weighted_slot_counts[slot] += weight*count
                weighted_lemma_counts[lemma] += weight*count
                weighted_pos_counts[pos] += weight*count

    joint_topic_count = sum(topic_term_counts.values())

    if topic_id:
        if stem_file:
            append_topic_morphological_analysis_to_file(topic_id, stem_file, weighted_slot_counts, joint_topic_count)
        if lemma_file:
            append_topic_morphological_analysis_to_file(topic_id, lemma_file, weighted_lemma_counts, joint_topic_count)
        if pos_file:
            append_topic_morphological_analysis_to_file(topic_id, pos_file, weighted_pos_counts, joint_topic_count)

    slot_probs = np.array(list(weighted_slot_counts.values())) / joint_topic_count
    pos_probs = np.array(list(weighted_pos_counts.values())) / joint_topic_count
    lemma_probs = np.array(list(weighted_lemma_counts.values())) / joint_topic_count

    slot_entropy = entropy(slot_probs)
    pos_entropy = entropy(pos_probs)
    lemma_entropy = entropy(lemma_probs)

    return slot_entropy, len(weighted_slot_counts), pos_entropy, len(weighted_pos_counts), lemma_entropy, len(weighted_lemma_counts)


def morphological_entropy_all_topics(in_state_file, stem_analysis_file=None, lemma_analysis_file=None, pos_analysis_file=None):
    """Computes entropy by various levels of morphological analysis

    :param in_state_file: Mallet .gzip file produced by mallet train-topics --output-state option
    :returns: pandas DataFrame
    """
    # prep detailed analysis files
    for (f, col) in [(stem_analysis_file, "stem"), (lemma_analysis_file, "lemma"), (pos_analysis_file, "pos")]:
        if f:
            write_header_to_morph_analysis_file(f, col)

    gzip_reader = gzip.open(in_state_file, mode='rt', encoding='utf-8')

    header = gzip_reader.readline()
    alpha_text = gzip_reader.readline()
    alphas = [float(a) for a in alpha_text.strip().split(' : ')[1].split()]
    n_topics = len(alphas)
    beta_text = gzip_reader.readline()

    topic_term_counts = {i:Counter() for i in range(n_topics)}

    current_doc_id = -1
    for i, line in enumerate(gzip_reader):
        fields = line.strip().split()
        doc_id = int(fields[0])
        term = fields[4]
        topic_id = int(fields[5])
        if doc_id and doc_id % 1000==0:
            if doc_id != current_doc_id:
                print("Reading terms for doc:", doc_id)
                current_doc_id = doc_id

        topic_term_counts[topic_id][term] += 1

    gzip_reader.close()

    mystem = pymystem3.Mystem(disambiguation=False)

    result = []

    # Determine P(slot|k) for each topic
    for topic_id in topic_term_counts:
        if topic_id % 10 == 0:
            print("Calculating entropies for topic:", topic_id)

        slot_entropy, num_slots, pos_entropy, num_pos, lemma_entropy, num_lemmas = morphological_entropy_single_topic(mystem, topic_term_counts[topic_id], topic_id, stem_analysis_file, lemma_analysis_file, pos_analysis_file)

        result.append([topic_id, slot_entropy, num_slots, pos_entropy, num_pos, lemma_entropy, num_lemmas])

    return pd.DataFrame.from_records(result, columns=['topic_id', 'slot_entropy', 'num_slots', 'pos_entropy', 'num_pos', 'lemma_entropy', 'num_lemmas'])


def stem_state_file(in_state_file, out_state_file, stemmer):
    """Creates a new Mallet .gzip state file by post processing vocab elements with stemming.

    :param in_state_file: Mallet .gzip file produced by mallet train-topics --output-state option
    :param out_state_file: Target path for new .gzip file
    :param vocab_index: map of stemmed vocab item to int index
    :param stemmer: a Stemmer object from topic_modeling.stemming
    :returns: Map of topic id to a Counter with stemmed terms as keys and counts
        of the stem for that topic
    """
    # TODO Some clean up needed here
    # TODO What do we want to do with topic/term counts?
    vocab_index = {}
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

    topic_term_counts = {i:Counter() for i in range(n_topics)}

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
        if stemmed_term != '' and stemmed_term not in vocab_index:
            stemmed_term_index = len(vocab_index)
            vocab_index[stemmed_term] = stemmed_term_index
        else:
            stemmed_term_index = vocab_index[stemmed_term]

        output_result = " ".join([str(doc_id), source, pos, str(stemmed_term_index), stemmed_term, str(topic)])
        gzip_writer.write(output_result + "\n")

        topic_term_counts[topic][stemmed_term_index] += 1

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
state_file_parser.add_argument('out_gz', help="Desired path for new gzip to be created")
state_file_parser.add_argument('--lemmatizer', '-l',
    help='Choice of stemmer/lemmatizer',
    choices=stemming.STEMMER_CHOICES)

slot_entropy_parser = subparsers.add_parser('slot-entropy', help="Produces 'slot entropy' values for each topic given a Mallet state file")
slot_entropy_parser.add_argument('in_gz', help="Input gzip file, an existing state file")
slot_entropy_parser.add_argument('out_tsv', help="Target path for morphological entropy metrics in TSV format")
slot_entropy_parser.add_argument("--stem-analysis", "-s", help="Target path for detailed stem metrics by topic")
slot_entropy_parser.add_argument("--lemma-analysis", "-l", help="Target path for detailed lemma metrics by topic")
slot_entropy_parser.add_argument("--pos-analysis", "-p", help="Target path for detailed pos metrics by topic")


if __name__ == "__main__":
    args = parser.parse_args()
    subparser_name = args.subparser_name
    if subparser_name=="state-file":
        stemmer = stemming.pick_lemmatizer(args.lemmatizer)
        print("Producing stemmed version of", args.in_gz, "to be written to", args.out_gz)
        topic_term_counts = stem_state_file(args.in_gz, args.out_gz, stemmer)
    elif subparser_name=="slot-entropy":
        print("Determining morphological slot entropy for topics in", args.in_gz)
        slot_entropy_df = morphological_entropy_all_topics(args.in_gz, args.stem_analysis, args.lemma_analysis, args.pos_analysis)
        print("Writing resulting dataframe to", args.out_tsv)
        slot_entropy_df.to_csv(args.out_tsv, sep="\t", index=False)

    else:
        diagnostics_df = diagnostics_xml_to_dataframe(args.in_xml)
        # Make sure pandas will print everything
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', -1)
        print(diagnostics_df.describe())
        diagnostics_df.to_csv(args.out_tsv, sep="\t")
