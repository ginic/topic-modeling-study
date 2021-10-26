"""Functions for dealing with Mallet's diagnostics and state files.
"""
# TODO Change to used oracle gzip file for morphology analysis instead of mystem
# TODO Remove weighted entropy calculations (this is a mystem thing)
import argparse
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
import gzip

import numpy as np
from numpy.core.numeric import full
import pandas as pd
import pymystem3

import topic_modeling.stemming as stemming

# Diagnostics XML constants
TOPIC="topic"
TOPIC_ID="id"
DEFAULT_TOP_N=20
UNKNOWN="UNKNOWN"

# Column names for dataframes and grouping
TOPIC_KEY = "topic"
TERM_KEY = "term"
SLOT_KEY = "slot"
LEMMA_KEY = "lemma"
POS_KEY = "pos"
DOC_ID_KEY = "doc_id"
PROB_KEY = "conditional_probability"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


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


def morphological_entropy_single_topic(topic_term_counts, topic_id=None, slot_file=None, lemma_file=None, pos_file=None, top_n=DEFAULT_TOP_N):
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


        top_n_counter += 1

    joint_topic_count = sum(topic_term_counts.values())

    # Calculates all metrics we want to write to file
    unweighted_slot_entropy = get_entropy_from_counts_dict(unweighted_slot_counts, joint_topic_count)
    unweighted_pos_entropy = get_entropy_from_counts_dict(unweighted_pos_counts, joint_topic_count)
    unweighted_lemma_entropy = get_entropy_from_counts_dict(unweighted_lemma_counts, joint_topic_count)

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


    return unweighted_slot_entropy, len(unweighted_slot_counts), unweighted_pos_entropy, len(unweighted_pos_counts), unweighted_lemma_entropy, len(unweighted_lemma_counts), unweighted_ratio_slots_lemmas, unweighted_ratio_pos_lemmas, lemmas_to_forms_top_n, slots_to_forms_top_n, pos_to_forms_top_n,  top_n_coverage


def morphological_entropy_all_topics(in_state_file, oracle_file, ):
    """Reads in Mallet state file and the oracle file produced from corpus_preprocessing and returns
    a pandas DataFrame with topic,term,lemma,slot,POS tag as columns from which entropy metrics can be computed.

    :param in_state_file: Mallet Gzip file produced by mallet train-topics --output-state option
    :param oracle_file: Gzipped TSV file produced by corpus_preprocessing.py that contains the gold standard lemma and morphological analysis from the corpus
    :returns: pandas DataFrame
    """
    gzip_reader = gzip.open(in_state_file, mode='rt', encoding='utf-8')

    # Burn header
    header = gzip_reader.readline()
    alpha_text = gzip_reader.readline()
    alphas = [float(a) for a in alpha_text.strip().split(' : ')[1].split()]
    n_topics = len(alphas)
    # Burn beta values
    beta_text = gzip_reader.readline()

    # Oracle reader should now exactly line up with the Mallet file
    oracle_reader = gzip.open(oracle_file, mode='rt', encoding='utf-8')

    # Collate everything into lists that are put as columns in a pandas DF
    topics = []
    terms = []
    lemmas = []
    slots = []
    parts_of_speech = []
    doc_ids = []

    current_doc_id = -1
    for _, line in enumerate(gzip_reader):
        _, oracle_doc_idx, surface_form, lemma, morph_analysis = oracle_reader.readline().strip().split()
        oracle_doc_idx = int(oracle_doc_idx)
        pos_tag = morph_analysis.split(',')[0]
        fields = line.strip().split()
        doc_id = int(fields[0])
        term = fields[4]
        topic_id = int(fields[5])
        # Verify document ids are lining up
        assert oracle_doc_idx == doc_id, f"Mallet file id {doc_id} (term {term}) not lining up with oracle doc id {oracle_doc_idx} (surface form {surface_form})"

        if doc_id and doc_id % 1000==0:
            if doc_id != current_doc_id:
                print("Reading terms for doc:", doc_id)
                current_doc_id = doc_id

        topics.append(topic_id)
        terms.append(term)
        lemmas.append(lemma)
        slots.append(morph_analysis)
        parts_of_speech.append(pos_tag)
        doc_ids.append(doc_id)

    gzip_reader.close()
    oracle_reader.close()

    analysis_df = pd.DataFrame.from_dict(
        {TOPIC_KEY:topics,
        TERM_KEY:terms,
        LEMMA_KEY:lemmas,
        SLOT_KEY:slots,
        POS_KEY:parts_of_speech,
        DOC_ID_KEY:doc_ids})
    return analysis_df

def compute_top_n_metrics(parsed_topic_df, top_n=DEFAULT_TOP_N):
    """Computes the number of unique slots, lemmas and POS tags in the top_n terms for each topic.
    Returns the result as a pandas DataFrame with columns: topic, lemma_entropy, slot_entropy, pos_entropy
    :param parsed_topic_df: Pandas DataFrame containing the topic,term,lemma,slot,pos,doc_ids for each word in the corpus
    :param top_n:  int, the number of terms to consider for building ratios based on top n word forms in topic
    """
    # Count number of grammatical forms in top_n most common terms for each topic
    term_counts = parsed_topic_df.groupby([TOPIC_KEY, TERM_KEY]).size().reset_index()
    term_counts.columns = [TOPIC_KEY, TERM_KEY, 'term_count']
    # Sort by term count, then take top n within each group
    sort_by_terms = term_counts.groupby(TOPIC_KEY).apply(lambda x: x.sort_values(['term_count'], ascending=False)).reset_index(drop=True)
    top_terms = sort_by_terms.groupby(TOPIC_KEY).head(top_n)
    top_terms.to_csv('top_terms.tsv', sep='\t')
    filter_by_top_terms = pd.merge(top_terms, parsed_topic_df, on=[TOPIC_KEY, TERM_KEY])
    filter_by_top_terms.to_csv('filter_by_top_terms.tsv', sep='\t')
    final_top_terms = (filter_by_top_terms.groupby(TOPIC_KEY)
        .agg({LEMMA_KEY:'nunique', SLOT_KEY:'nunique', POS_KEY:'nunique'})
        .rename(columns={LEMMA_KEY:f'lemmas_to_top_{top_n}_surface_forms', SLOT_KEY:f'slots_to_top_{top_n}_surface_forms', POS_KEY:f'pos_to_top_{top_n}_surface_forms'})
        .reset_index()
    )
    return final_top_terms


def compute_entropy_metrics(parsed_topic_df, slot_analysis_file=None, lemma_analysis_file=None, pos_analysis_file=None):
    """Computes entropies for each topic, writing results to file as desired.
    Returns the results as a pandas DataFrame with columns: topic, lemmas_to_top_n_surface_forms, slots_to_top_n_surface_forms, pos_to_top_n_surface_forms

    :param parsed_topic_df: Pandas DataFrame containing the topic,term,lemma,slot,pos,doc_ids for each word in the corpus
    :param slot_analysis_file: optional file path to write detailed slot metrics to
    :param lemma_analysis_file: optional file path to write detailed lemma metrics to
    :param pos_analyais_file: optional file path to write detailed POS metrics to
    :param top_n:  int, the number of terms to consider for building ratios based on top n word forms in topic
    """
    marginal_count_key = "topic_marginal_count"
    topic_marginal_count = parsed_topic_df.groupby(TOPIC_KEY, as_index=False).size().rename(columns={'size': marginal_count_key})
    full_results = topic_marginal_count

    # Get joint counts and compute entropy for each type of grammatical information
    for k, out_file in [(LEMMA_KEY, lemma_analysis_file), (SLOT_KEY, slot_analysis_file), (POS_KEY, pos_analysis_file)]:
        joint_col = f'{k}_topic_joint_count'
        joint_counts = parsed_topic_df.groupby([TOPIC_KEY, k], as_index=False).size().sort_values(by='size', ascending=False).rename(columns={'size':joint_col})
        joined_table_for_entropy = pd.merge(topic_marginal_count, joint_counts, on=TOPIC_KEY)
        # Compute and add entropy for each topic to full results table
        joined_table_for_entropy[PROB_KEY] = joined_table_for_entropy[joint_col] / joined_table_for_entropy[marginal_count_key]
        if out_file:
            joined_table_for_entropy.to_csv(out_file, sep="\t", index=False, columns = [TOPIC_KEY, k, joint_col, PROB_KEY])
        topic_entropy = joined_table_for_entropy.groupby(TOPIC_KEY).agg({PROB_KEY: entropy}).rename(columns={PROB_KEY:f'{k}_entropy'})
        full_results = pd.merge(full_results, topic_entropy, on=TOPIC_KEY)

    return full_results.drop(columns = marginal_count_key)


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

# TODO Expand to support other languages
state_file_parser = subparsers.add_parser('post-stem', help="Perform post-stemming of a topic model on the Mallet state file")
state_file_parser.add_argument('in_gz', help="Input gzip file, an existing state file")
state_file_parser.add_argument('out_gz', help="Desired path for new gzip to be created")
state_file_parser.add_argument('--lemmatizer', '-l',
    help='Choice of stemmer/lemmatizer',
    choices=stemming.STEMMER_CHOICES)

slot_entropy_parser = subparsers.add_parser('slot-entropy', help="Produces 'slot entropy' values for each topic given a Mallet state file")
slot_entropy_parser.add_argument('in_gz', help="Input gzip file, an existing state file")
slot_entropy_parser.add_argument('oracle_gz', help="Gzipped TSV that contains oracle forms and analysis for a corpus. This file is produced by corpus_preprocessing.py")
slot_entropy_parser.add_argument('out_tsv', help="Target path for morphological entropy metrics in TSV format")
slot_entropy_parser.add_argument("--slot-analysis", "-s", help="Target path for detailed slot metrics by topic")
slot_entropy_parser.add_argument("--lemma-analysis", "-l", help="Target path for detailed lemma metrics by topic")
slot_entropy_parser.add_argument("--pos-analysis", "-p", help="Target path for detailed pos metrics by topic")


if __name__ == "__main__":
    args = parser.parse_args()
    subparser_name = args.subparser_name
    if subparser_name=="post-stem":
        # TODO post-stemming could also be done using the oracle file
        stemmer = stemming.pick_lemmatizer(args.lemmatizer)
        print("Producing stemmed version of", args.in_gz, "to be written to", args.out_gz)
        topic_term_counts = stem_state_file(args.in_gz, args.out_gz, stemmer)
    elif subparser_name=="slot-entropy":
        print("Determining morphological slot entropy for topics in", args.in_gz, "using oracle", args.oracle_gz)
        topic_analysis_df = morphological_entropy_all_topics(args.in_gz, args.oracle_gz)
        print("Parsed dataframe from gzips, head:")
        print(topic_analysis_df.head())
        entropy_metrics = compute_entropy_metrics(topic_analysis_df, args.slot_analysis, args.lemma_analysis, args.pos_analysis)
        print("Calculated entropy metrics, head:")
        print(entropy_metrics.head())
        top_n_metrics = compute_top_n_metrics(topic_analysis_df)
        print("Calculated top N term metrics, head:")
        print(top_n_metrics.head())
        full_metrics_df = pd.merge(entropy_metrics, top_n_metrics , on=TOPIC_KEY)
        print("Writing resulting dataframe to", args.out_tsv)
        full_metrics_df.to_csv(args.out_tsv, sep="\t", index=False)

    else:
        diagnostics_df = diagnostics_xml_to_dataframe(args.in_xml)
        diagnostics_df['negative_coherence'] = - diagnostics_df['coherence']
        # Make sure pandas will print everything
        print(diagnostics_df.describe())
        diagnostics_df.to_csv(args.out_tsv, sep="\t")
