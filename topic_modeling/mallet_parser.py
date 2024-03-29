"""Functions for dealing with Mallet's diagnostics and state files.
"""
# TODO Change to used oracle gzip file for morphology analysis instead of mystem
import argparse
from collections import Counter, defaultdict
from typing_extensions import final
import xml.etree.ElementTree as ET
import gzip
import os
import re

import numpy as np
import pandas as pd

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
PROB_KEY = "conditional_probability_given_topic"

# Part of speech can be separated by comma or equal, depending on the corpus
POS_SPLIT_PATTERN = re.compile('[,=]')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def get_state_file_from_surface_forms(raw_state_file, treatment_state_file, output_state_file):
    """Creates a state file for a topic model that uses the same topic allocations as the treatment model, but the surface forms from the raw state file as terms/vocabulary. The raw state file and treatment state file must underlyingly have the same documents and tokens, but vocabularly may differ due to stemming/lemmatization.

    This is used for mapping back to surface forms in order to compute coherence using the same untreated vocabulary.

    :param raw_state_file: path to the state file using the terms you want to have in output
    :param treatment_state_file: path to state file for model weights and allocations you want in output
    :param output_state_file: path to desired output state file
    """
    raw_reader = gzip.open(raw_state_file, mode='rt', encoding='utf-8')
    treatment_reader = gzip.open(treatment_state_file, mode='rt', encoding='utf-8')

    # skip headers and weights in raw model - 3 lines
    for i in range(3):
     _ = raw_reader.readline()

    # write headers and weights from treatment model
    gzip_writer = gzip.open(output_state_file, mode='wt', encoding='utf-8')
    header = treatment_reader.readline()
    gzip_writer.write(header)
    alpha_text = treatment_reader.readline()
    gzip_writer.write(alpha_text)
    beta_text = treatment_reader.readline()
    gzip_writer.write(beta_text)

    # The treament and raw state file should be the same length
    count=0
    for treatment_line in treatment_reader:
        # topic allocation comes from the treatment model, everything else from the raw
        _, _, _, _, treatment_term, treatment_topic = split_mallet_state_file_row(treatment_line)
        doc_id, source, pos, term_idx, term, _ = split_mallet_state_file_row(raw_reader.readline())
        output_result = " ".join([str(doc_id), source, pos, term_idx, term, str(treatment_topic)])
        gzip_writer.write(output_result + "\n")
        if count % 10000==0:
            print("Document:", doc_id, "Raw term:", term, "Treated term:", treatment_term)
        count+=1

    treatment_reader.close()
    raw_reader.close()
    gzip_writer.close()


def get_corpus_analysis_counts(oracle_file):
    """Computes joint counts between word type, lemmas and word type, slots and word type, pos tag.
    Returns results as tuples of (surface_form_index, analysis_index, np array freq matrix).
    :param oracle_file: Path to the oracleAnalysis gzip file
    """
    oracle_reader = gzip.open(oracle_file, mode='rt', encoding='utf-8')
    surface_form_counter = Counter()
    lemma_counter = Counter()
    slot_counter = Counter()
    pos_counter = Counter()

    for line in oracle_reader:
        _, _, surface_form, lemma, morph_analysis, pos_tag = split_oracle_gz_row(line)
        surface_form_counter[surface_form] += 1
        lemma_counter[lemma] += 1
        slot_counter[morph_analysis] += 1
        pos_counter[pos_tag] += 1

    return surface_form_counter, lemma_counter, slot_counter, pos_counter


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


def parse_morphological_analysis_all_topics(in_state_file, oracle_file):
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
    for line in gzip_reader:
        _, oracle_doc_idx, surface_form, lemma, morph_analysis, pos_tag = split_oracle_gz_row(oracle_reader.readline())
        doc_id, _, _, _, term, topic_id = split_mallet_state_file_row(line)
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

def compute_top_n_metrics(parsed_topic_df, top_terms_file, top_n=DEFAULT_TOP_N ):
    """Computes the number of unique slots, lemmas and POS tags in the top_n terms for each topic.
    Returns the result as a pandas DataFrame with columns: topic, lemmas_to_top_surface_forms, slots_to_top_surface_forms, pos_to_top_surface_forms, top_n_term_set, top_n_lemma_set, lemmas_in_top
    :param parsed_topic_df: Pandas DataFrame containing the topic,term,lemma,slot,pos,doc_ids for each word in the corpus
    :param top_n:  int, the number of terms to consider for building ratios based on top n word forms in topic
    """

    # Collect top terms for each topic
    top_n_term_counts = (parsed_topic_df.groupby([TOPIC_KEY, TERM_KEY])
        .size()
        .reset_index())
    top_n_term_counts.columns = [TOPIC_KEY, TERM_KEY, 'term_count']
    # Sort by term count, then take top n within each group
    top_n_term_counts = (top_n_term_counts.groupby(TOPIC_KEY)
        .apply(lambda x: x.nlargest(top_n, 'term_count'))
        .reset_index(drop=True))
    top_n_term_counts.to_csv(top_terms_file, sep='\t', index=False)
    print("Top n term counts")
    print(top_n_term_counts.head())

    print("Top term sets:")
    top_term_sets = (top_n_term_counts.groupby(TOPIC_KEY)
        .agg({TERM_KEY: [(f'top_{top_n}_term_set', set)]})
        .droplevel(0, axis=1))
    print(top_term_sets.head())

    # Collect top n lemmas for each topic set
    print("LEMMA COUNTS")
    top_n_lemma_counts = (parsed_topic_df.groupby([TOPIC_KEY, LEMMA_KEY])
        .size()
        .reset_index())
    top_n_lemma_counts.columns = [TOPIC_KEY, LEMMA_KEY, 'lemma_count']
    top_n_lemma_counts = (top_n_lemma_counts.groupby(TOPIC_KEY)
        .apply(lambda x: x.nlargest(top_n, 'lemma_count'))
        .reset_index(drop=True))
    print(top_n_lemma_counts.head())

    print("Top N Lemma sets:")
    top_lemma_sets = (top_n_lemma_counts.groupby(TOPIC_KEY)
        .agg({LEMMA_KEY: [(f'top_{top_n}_lemma_set', set)]})
        .droplevel(0, axis=1))
    print(top_lemma_sets.head())

    # Determine the set of lemmas related to the top n terms
    filter_by_top_terms = pd.merge(top_n_term_counts, parsed_topic_df, on=[TOPIC_KEY, TERM_KEY])
    lemmas_in_top_terms = (filter_by_top_terms.groupby(TOPIC_KEY)
        .agg({LEMMA_KEY: [(f'lemmas_in_top_{top_n}_terms', set)]})
        .droplevel(0, axis=1))
    print("Lemmas in top terms set:")
    print(lemmas_in_top_terms.head())

    # Collect up metrics about grammatical forms in the top n terms
    filter_by_top_terms.to_csv('filter_by_top_terms.tsv', sep='\t')
    final_top_terms = (filter_by_top_terms.groupby(TOPIC_KEY)
        .agg({LEMMA_KEY: 'nunique', SLOT_KEY: 'nunique', POS_KEY: 'nunique'})
        .rename(columns={LEMMA_KEY:f'lemmas_to_top_{top_n}_surface_forms', SLOT_KEY:f'slots_to_top_{top_n}_surface_forms', POS_KEY:f'pos_to_top_{top_n}_surface_forms'})
        .reset_index())
    for k in [f'lemmas_to_top_{top_n}_surface_forms', f'slots_to_top_{top_n}_surface_forms', f'pos_to_top_{top_n}_surface_forms']:
        final_top_terms[k] = final_top_terms[k] / top_n

    # Merge in all term and lemma sets
    for set_df in [top_term_sets, top_lemma_sets, lemmas_in_top_terms]:
        final_top_terms = pd.merge(final_top_terms, set_df, on=TOPIC_KEY)

    # Computations using sets
    final_top_terms['top_lemmas_minus_top_term_lemmas'] = final_top_terms[f'top_{top_n}_lemma_set'] - final_top_terms[f'lemmas_in_top_{top_n}_terms']
    final_top_terms['top_term_lemmas_minus_top_lemmas'] = final_top_terms[f'lemmas_in_top_{top_n}_terms'] - final_top_terms[f'top_{top_n}_lemma_set']
    final_top_terms[f'num_lemmas_in_top_{top_n}_terms'] = final_top_terms[f'lemmas_in_top_{top_n}_terms'].apply(len)
    final_top_terms['num_top_lemmas_excluded_by_top_terms'] = final_top_terms['top_lemmas_minus_top_term_lemmas'].apply(len)
    final_top_terms['num_top_term_lemmas_excluded_by_top_lemmas'] = final_top_terms['top_term_lemmas_minus_top_lemmas'].apply(len)

    # Convert sets to space separated lists for
    join_set = lambda x: " ".join(x) if len(x)> 0 else ""
    final_top_terms['top_lemmas_minus_top_term_lemmas'] = final_top_terms['top_lemmas_minus_top_term_lemmas'].apply(join_set)
    final_top_terms['top_term_lemmas_minus_top_lemmas'] = final_top_terms['top_term_lemmas_minus_top_lemmas'].apply(join_set)
    final_top_terms[f'lemmas_in_top_{top_n}_terms'] = final_top_terms[f'lemmas_in_top_{top_n}_terms'].apply(join_set)
    final_top_terms[f'top_{top_n}_lemma_set'] = final_top_terms[f'top_{top_n}_lemma_set'].apply(join_set)
    final_top_terms[f'top_{top_n}_term_set'] = final_top_terms[f'top_{top_n}_term_set'].apply(join_set)

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
    topic_marginal_count = parsed_topic_df.groupby(TOPIC_KEY, as_index=False).count().rename(columns={TERM_KEY: marginal_count_key})
    topic_marginal_count = topic_marginal_count[[TOPIC_KEY, marginal_count_key]]
    full_results = topic_marginal_count

    # Get joint counts and compute entropy for each type of grammatical information
    for k, out_file in [(LEMMA_KEY, lemma_analysis_file), (SLOT_KEY, slot_analysis_file), (POS_KEY, pos_analysis_file)]:
        joint_col = f'{k}_topic_joint_count'
        joint_counts = parsed_topic_df.groupby([TOPIC_KEY, k], as_index=False).count()[[TOPIC_KEY, k, TERM_KEY]]
        joint_counts = joint_counts.rename(columns={TERM_KEY:joint_col}).sort_values(by=[TOPIC, joint_col], ascending=[True, False])
        joined_table_for_entropy = pd.merge(topic_marginal_count, joint_counts, on=TOPIC_KEY)
        # Compute and add entropy for each topic to full results table
        joined_table_for_entropy[PROB_KEY] = joined_table_for_entropy[joint_col] / joined_table_for_entropy[marginal_count_key]

        if out_file:
            joined_table_for_entropy.to_csv(out_file, sep="\t", index=False, columns = [TOPIC_KEY, k, joint_col, PROB_KEY])
        topic_entropy = joined_table_for_entropy.groupby(TOPIC_KEY).agg({PROB_KEY: entropy}).rename(columns={PROB_KEY:f'{k}_entropy'})
        full_results = pd.merge(full_results, topic_entropy, on=TOPIC_KEY)

    return full_results.drop(columns = marginal_count_key)


def split_mallet_state_file_row(mallet_tsv_row):
    """Parses out the appropriate columns from a Mallet Topic Model state file
    Returns doc_id (as int), source, position, term, topic (as int)
    :param mallet_tsv_row: str, line from the Mallet state file
    """
    fields = mallet_tsv_row.strip().split()
    doc_id = int(fields[0])
    source = fields[1]
    pos = fields[2]
    term_idx = fields[3]
    term = fields[4]
    topic = int(fields[5])
    return doc_id, source, pos, term_idx, term, topic


def split_oracle_gz_row(oracle_row, use_downcase=True):
    """Parses out oracle morphological information from a row in the oracle tsv.
    Returns the document name, document_id (as int, matches Mallet doc_id), surface form of the word, the lemma, the morphological analysis from the corpus and the part of speech tag
    :param oracle_row: str, line from the oracle TSV
    :param use_downcase: True to downcase the surface form and lemma
    """
    doc_name, oracle_doc_idx, surface_form, lemma, morph_analysis = oracle_row.strip().split()
    oracle_doc_idx = int(oracle_doc_idx)
    pos_tag = POS_SPLIT_PATTERN.split(morph_analysis)[0]
    if use_downcase:
        lemma = lemma.lower()
        surface_form = surface_form.lower()
    return doc_name, oracle_doc_idx, surface_form, lemma, morph_analysis, pos_tag


def stem_state_file(in_state_file, out_state_file, stemmer=None, oracle_gz=None, use_downcase=True):
    """Creates a new Mallet .gzip state file by post processing vocab elements with stemming.

    :param in_state_file: Mallet .gzip file produced by mallet train-topics --output-state option
    :param out_state_file: Target path for new .gzip file
    :param stemmer: a Stemmer object from topic_modeling.stemming
    :param oracle_gz: The oracle file produced by corpus_preprocessor.py that is token aligned with the Mallet state files.
    :param use_downcase: True to downcase lemmatized/stemmed types before writing them to the Mallet state file. Generally, you want to do this.
    :returns: Map of topic id to a Counter with stemmed terms as keys and counts of the stem for that topic
    """
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

    oracle_reader = None
    if oracle_gz:
        oracle_reader = gzip.open(oracle_gz, mode='rt', encoding='utf-8')

    topic_term_counts = {i:Counter() for i in range(n_topics)}

    current_doc_id = -1
    for line in gzip_reader:
        doc_id, source, pos, _, term, topic = split_mallet_state_file_row(line)
        if doc_id and doc_id % 1000==0:
            if doc_id != current_doc_id:
                print("Reading terms for doc:", doc_id)
                current_doc_id = doc_id

        # Using stemmer
        if stemmer:
            stemmed_term = stemmer.single_term_lemma(term)
        # Use oracle
        elif oracle_reader:
            _, oracle_doc_idx, _, lemma, _, _ = split_oracle_gz_row(oracle_reader.readline(), use_downcase)
            # Models still use downcase version of lemmas
            stemmed_term = lemma
            # Verify document ids are lining up
            assert oracle_doc_idx == doc_id, f"Mallet file id {doc_id} (term {term}) not lining up with oracle doc id {oracle_doc_idx}"

        if use_downcase:
            stemmed_term = stemmed_term.lower()

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

diagnostics_parser = subparsers.add_parser('diagnostics', help="Parses Mallet diagnostics file to CSV and reports overall statistics")
diagnostics_parser.add_argument('in_xml', help="Mallet diagnostics xml file")
diagnostics_parser.add_argument('out_tsv', help="Target path for diagnostics in TSV format")

state_file_parser = subparsers.add_parser('post-stem', help="Perform post-stemming of a topic model on the Mallet state file")
state_file_parser.add_argument('in_gz', help="Input gzip file, an existing state file")
state_file_parser.add_argument('out_gz', help="Desired path for new gzip for new model state file to be created")
state_file_parser.add_argument('--lemmatizer', '-l',
    help='Choice of stemmer/lemmatizer',
    choices=stemming.STEMMER_CHOICES)

oracle_post_stem_parser = subparsers.add_parser('oracle-post-lemmatize', help="Perform post stemming of a topic model using the oracle lemmas from a gzip TSV file")
oracle_post_stem_parser.add_argument('in_gz', help="Input gzip file, an existing state file")
oracle_post_stem_parser.add_argument('out_gz', help="Desired path for new gzip for new model state file to be created")
oracle_post_stem_parser.add_argument('oracle_gz', help="Gzipped TSV that contains oracle forms and analysis for a corpus. This file is produced by corpus_preprocessing.py")

slot_entropy_parser = subparsers.add_parser('slot-entropy', help="Produces 'slot entropy' values for each topic given a Mallet state file")
slot_entropy_parser.add_argument('in_gz', help="Input gzip file, an existing state file")
slot_entropy_parser.add_argument('oracle_gz', help="Gzipped TSV that contains oracle forms and analysis for a corpus. This file is produced by corpus_preprocessing.py")
slot_entropy_parser.add_argument('out_dir', help="Target directory for output TSVs")
slot_entropy_parser.add_argument('out_prefix', help="Prefix to add to output TSV files to help track them better")

unmap_stemming_parser = subparsers.add_parser('unmap-stemmed-state', help="Map back to original raw terms in order to compute coherence metrics using the same vocabulary")
unmap_stemming_parser.add_argument('raw_gz', help="Gzip file for the Mallet state file to use terms/vocabulary from")
unmap_stemming_parser.add_argument('treatment_gz', help="Gzip file for Mallet state file from which to use topic assignments for tokens")
unmap_stemming_parser.add_argument('output_gz', help="Path to the new gzip Mallet state file that will be created")



if __name__ == "__main__":
    args = parser.parse_args()
    subparser_name = args.subparser_name
    if subparser_name=="post-stem":
        # TODO post-stemming could also be done using the oracle file
        stemmer = stemming.pick_lemmatizer(args.lemmatizer)
        print("Producing stemmed version of", args.in_gz, "to be written to", args.out_gz)
        topic_term_counts = stem_state_file(args.in_gz, args.out_gz, stemmer=stemmer)
    elif subparser_name=="oracle-post-lemmatize":
        print("Producing post-lemmatized version of topic model from the oracle lemmas")
        stem_state_file(args.in_gz, args.out_gz, oracle_gz=args.oracle_gz)
    elif subparser_name=="slot-entropy":
        print("Determining morphological slot entropy for topics in", args.in_gz, "using oracle", args.oracle_gz)
        topic_analysis_df = parse_morphological_analysis_all_topics(args.in_gz, args.oracle_gz)
        print("Parsed dataframe from gzips, head:")
        print(topic_analysis_df.head())

        slot_analysis_file = os.path.join(args.out_dir, args.out_prefix + "_topic_slots.tsv")
        lemma_analysis_file = os.path.join(args.out_dir, args.out_prefix + "_topic_lemmas.tsv")
        pos_analysis_file = os.path.join(args.out_dir, args.out_prefix + "_topic_pos.tsv")
        entropy_file = os.path.join(args.out_dir, args.out_prefix + "_entropy_metrics.tsv")
        entropy_metrics = compute_entropy_metrics(topic_analysis_df, slot_analysis_file, lemma_analysis_file, pos_analysis_file)
        print("Calculated entropy metrics, head:")
        print(entropy_metrics.head())

        top_terms_csv = os.path.join(args.out_dir, args.out_prefix + "_top_terms.tsv")
        top_n_metrics = compute_top_n_metrics(topic_analysis_df, top_terms_csv)
        print("Calculated top N term metrics, head:")
        print(top_n_metrics.head())
        full_metrics_df = pd.merge(entropy_metrics, top_n_metrics , on=TOPIC_KEY)
        print("Writing resulting dataframe to", entropy_file)
        full_metrics_df.to_csv(entropy_file, sep="\t", index=False)
    elif subparser_name=="unmap-stemmed-state":
        print("Starting unstemming:")
        print("Raw state file:", args.raw_gz)
        print("Treatment state file:", args.treatment_gz)
        print("Output state file:", args.output_gz)
        get_state_file_from_surface_forms(args.raw_gz, args.treatment_gz, args.output_gz)

    else:
        diagnostics_df = diagnostics_xml_to_dataframe(args.in_xml)
        diagnostics_df['negative_coherence'] = - diagnostics_df['coherence']
        # Make sure pandas will print everything
        print(diagnostics_df.describe())
        diagnostics_df.to_csv(args.out_tsv, sep="\t")
