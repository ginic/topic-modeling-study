"""Functions for dealing with Mallet's diagnostics and state files.
"""
# TODO Change to used oracle gzip file for morphology analysis instead of mystem
# TODO Remove weighted entropy calculations (this is a mystem thing)
import argparse
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
import gzip

import numpy as np
import pandas as pd
import pymystem3

import topic_modeling.stemming as stemming

# Diagnostics XML constants
TOPIC="topic"
TOPIC_ID="id"
DEFAULT_TOP_N=20
UNKNOWN="UNKNOWN"


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
        f.write(f"topic\t{col_name}\tweighted_count\tweighted_proportion\tunweighted_count\tunweighted_proportion\n")

def get_stemmed_vocab(vocab_file, stemmer):
    """Read in the vocabulary from a txt file, one element per line.
    Stems items and maps vocab appropriately to new indices
    Return list of vocab items and index mapping {term: index in list}.

    :param vocab_file: file in UTF-8 format, one term per line
    :param stemmer: a Stemmer object from topic_modeling.stemming
    """
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

def get_entropy_from_counts_dict(topic_counts_dict, total_term_count):
    """Calculates probabilities and entropy from a dict of counts for a term/lemma/pos/slot

    :param counts_dict: dict, str to float or int
    :param total_term_count: int, number of terms in topic overall (note, could also be retrieved from sum of in dict)
    """
    probs = np.array(list(topic_counts_dict.values())) / total_term_count
    return entropy(probs)

def append_topic_morphological_analysis_to_file(topic_id, tsv_file, weighted_morph_analysis_counts, unweighted_morph_analysis_counts, total_counts):
    """Writes out morphological analysis of a topic to tsv file in order of descending proportion
    """
    results = []
    for k,v in weighted_morph_analysis_counts.items():
        weighted_proportion = v / total_counts
        unweighted_count = unweighted_morph_analysis_counts[k]
        unweighted_proportion =  unweighted_count / total_counts
        results.append([topic_id, k, v, weighted_proportion, unweighted_count, unweighted_proportion])

    sorted_results = sorted(results, key=lambda x: x[3], reverse=True)

    with open(tsv_file, 'a') as f:
        for l in sorted_results:
            f.write("\t".join([str(x) for x in l]) + "\n")


def morphological_entropy_single_topic(mystem, topic_term_counts, topic_id=None, slot_file=None, lemma_file=None, pos_file=None, top_n=DEFAULT_TOP_N):
    """For a given topic, returns the slot entropy, number of slots, part of speech entropy, number of part of speech tags, lemma entropy, number of lemmas.
    If an outputdir is specified, the pos tags, slots and lemmas will be written

    :param mystem: Pymystem3 instance for determining mrophological analysis
    :param topic_term_counts: Counter {term: count} mapping for a single topic
    :param topic_id: int id of topic, only needed if you're going to write detailed results to a file
    :param slot_file: file to write detailed slot analysis to
    :param lemma_file: file to write detailed lemma analysis to
    :param pos_file: file to write detailed pos analysis to
    :param top_n_terms: int, the number of terms to consider for building ratios based on top n word forms in topic
    """
    # TODO This code is so cludgey, really needs to be reorged
    weighted_slot_counts = defaultdict(float)
    weighted_lemma_counts = defaultdict(float)
    weighted_pos_counts = defaultdict(float)
    unweighted_slot_counts = defaultdict(float)
    unweighted_lemma_counts = defaultdict(float)
    unweighted_pos_counts = defaultdict(float)
    lemmas_in_top_n_terms = defaultdict(float)
    slots_in_top_n_terms = defaultdict(float)
    pos_in_top_n_terms = defaultdict(float)

    top_n_counter = 0
    for surface_form, count in topic_term_counts.most_common():
        # We're just going to grab the first analysis element from pymystem3.
        # This might be wrong in some very small numbers of edge cases where pymystem3
        # tokenization is different (usually involves hyphens).
        analysis = mystem.analyze(surface_form)[0]

        # Pymystem returns no analysis, use UNKNOWN
        if len(analysis['analysis']) == 0:
            slot = UNKNOWN
            lemma = UNKNOWN
            pos=UNKNOWN
            unweighted_slot_counts[slot] += count
            unweighted_lemma_counts[lemma] += count
            unweighted_pos_counts[pos] += count
            weighted_slot_counts[slot] += count
            weighted_lemma_counts[lemma] += count
            weighted_pos_counts[pos] += count
            if top_n_counter < top_n:
                lemmas_in_top_n_terms[lemma] += count
                slots_in_top_n_terms[slot] += count
                pos_in_top_n_terms[pos] += count

        else:
            for i in range(len(analysis['analysis'])):
                morph_analysis = analysis['analysis'][i]
                slot = morph_analysis['gr']
                pos = slot.split("=")[0]
                weight = morph_analysis['wt']
                lemma = morph_analysis['lex']
                # For unweighted, just take the first result
                if i==0:
                    unweighted_slot_counts[slot] += count
                    unweighted_lemma_counts[lemma] += count
                    unweighted_pos_counts[pos] += count
                    if top_n_counter < top_n:
                        lemmas_in_top_n_terms[lemma] += count
                        slots_in_top_n_terms[slot] += count
                        pos_in_top_n_terms[pos] += count

                if weight != 0:
                    weighted_slot_counts[slot] += weight*count
                    weighted_lemma_counts[lemma] += weight*count
                    weighted_pos_counts[pos] += weight*count

        top_n_counter += 1

    joint_topic_count = sum(topic_term_counts.values())

    # Calculates all metrics we want to write to file
    weighted_slot_entropy = get_entropy_from_counts_dict(weighted_slot_counts, joint_topic_count)
    weighted_pos_entropy = get_entropy_from_counts_dict(weighted_pos_counts, joint_topic_count)
    weighted_lemma_entropy = get_entropy_from_counts_dict(weighted_lemma_counts, joint_topic_count)
    unweighted_slot_entropy = get_entropy_from_counts_dict(unweighted_slot_counts, joint_topic_count)
    unweighted_pos_entropy = get_entropy_from_counts_dict(unweighted_pos_counts, joint_topic_count)
    unweighted_lemma_entropy = get_entropy_from_counts_dict(unweighted_lemma_counts, joint_topic_count)

    weighted_ratio_slots_lemmas = len(weighted_slot_counts) / len(weighted_lemma_counts)
    weighted_ratio_pos_lemmas = len(weighted_pos_counts) / len(weighted_lemma_counts)
    unweighted_ratio_slots_lemmas = len(unweighted_slot_counts) / len(unweighted_lemma_counts)
    unweighted_ratio_pos_lemmas = len(unweighted_pos_counts) / len(unweighted_lemma_counts)
    lemmas_to_forms_top_n = len(lemmas_in_top_n_terms) / top_n
    slots_to_forms_top_n = len(slots_in_top_n_terms) / top_n
    pos_to_forms_top_n = len(pos_in_top_n_terms) / top_n
    top_n_coverage = sum(lemmas_in_top_n_terms.values()) / joint_topic_count

    # Write to file if desired
    if topic_id:
        if slot_file:
            append_topic_morphological_analysis_to_file(topic_id, slot_file, weighted_slot_counts, unweighted_slot_counts, joint_topic_count)
        if lemma_file:
            append_topic_morphological_analysis_to_file(topic_id, lemma_file, weighted_lemma_counts, unweighted_lemma_counts, joint_topic_count)
        if pos_file:
            append_topic_morphological_analysis_to_file(topic_id, pos_file, weighted_pos_counts, unweighted_pos_counts, joint_topic_count)


    return weighted_slot_entropy, len(weighted_slot_counts), weighted_pos_entropy, len(weighted_pos_counts), weighted_lemma_entropy, len(weighted_lemma_counts), unweighted_slot_entropy, len(unweighted_slot_counts), unweighted_pos_entropy, len(unweighted_pos_counts), unweighted_lemma_entropy, len(unweighted_lemma_counts), weighted_ratio_slots_lemmas, weighted_ratio_pos_lemmas, unweighted_ratio_slots_lemmas, unweighted_ratio_pos_lemmas, lemmas_to_forms_top_n, slots_to_forms_top_n, pos_to_forms_top_n,  top_n_coverage


def morphological_entropy_all_topics(in_state_file, slot_analysis_file=None, lemma_analysis_file=None, pos_analysis_file=None, top_n=DEFAULT_TOP_N):
    """Computes entropy by various levels of morphological analysis

    :param in_state_file: Mallet .gzip file produced by mallet train-topics --output-state option
    :param slot_analysis_file: optional file to write detailed slot metrics to
    :param lemma_analysis_file: optional file to write detailed lemma metrics to
    :param pos_analysis_file: optional file to write detailed pos metrics to
    :param top_n: number of top terms of each topic to consider for metrics that use it
    :returns: pandas DataFrame
    """
    # prep detailed analysis files
    for (f, col) in [(slot_analysis_file, "slot"), (lemma_analysis_file, "lemma"), (pos_analysis_file, "pos")]:
        if f:
            write_header_to_morph_analysis_file(f, col)

    gzip_reader = gzip.open(in_state_file, mode='rt', encoding='utf-8')

    # Burn header
    header = gzip_reader.readline()
    alpha_text = gzip_reader.readline()
    alphas = [float(a) for a in alpha_text.strip().split(' : ')[1].split()]
    n_topics = len(alphas)
    # Burn beta values
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

        results_tuple = morphological_entropy_single_topic(mystem, topic_term_counts[topic_id], topic_id, slot_analysis_file, lemma_analysis_file, pos_analysis_file)
        tmp_list = [topic_id]
        tmp_list.extend(results_tuple)
        result.append(tmp_list)

    return pd.DataFrame.from_records(result, columns=['topic_id', 'weighted_slot_entropy', 'num_weighted_slots', 'weighted_pos_entropy', 'num_weighted_pos', 'weighted_lemma_entropy', 'num_weighted_lemmas', 'unweighted_slot_entropy', 'num_unweighted_slots', 'unweighted_pos_entropy', 'num_unweighted_pos', 'unweighted_lemma_entropy', 'num_unweighted_lemmas', 'weighted_ratio_slots_to_lemmas', 'weighted_ratio_pos_to_lemmas','unweighted_ratio_slots_to_lemmas', 'unweighted_ratio_pos_to_lemmas', f'lemmas_to_top_{top_n}_surface_forms', f'slots_to_top_{top_n}_surface_forms', f'pos_to_top_{top_n}_surface_forms', f'topic_coverage_by_top_{top_n}_surface_forms'])


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
slot_entropy_parser.add_argument("--slot-analysis", "-s", help="Target path for detailed slot metrics by topic")
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
        slot_entropy_df = morphological_entropy_all_topics(args.in_gz, args.slot_analysis, args.lemma_analysis, args.pos_analysis)
        print("Writing resulting dataframe to", args.out_tsv)
        slot_entropy_df.to_csv(args.out_tsv, sep="\t", index=False)

    else:
        diagnostics_df = diagnostics_xml_to_dataframe(args.in_xml)
        diagnostics_df['negative_coherence'] = - diagnostics_df['coherence']
        # Make sure pandas will print everything
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(diagnostics_df.describe())
        diagnostics_df.to_csv(args.out_tsv, sep="\t")
