# coding=utf-8
'''Catch all module of classes and functions for analysis of term and document frequencies, generating by-author statistics, etc...

TODO: WIP as I figure out exactly what formats will work best for analysis
'''
import collections

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


import topic_modeling.tokenization as tokenization

# Tokenizer to use for analyzing vocab, terms and experiment results
TOKENIZER = tokenization.RegexTokenizer(tokenization.WORD_TYPE_NO_DIGITS_TOKENIZATION)

def get_token_list(token_file):
    """Returns a set of tokens from file where
    they're one per line. Useful for reading stop list or 'go' list.
    :param token_file: Path object
    """
    text = token_file.read_text(encoding="utf-8")
    tokens = set([s.strip() for s in text.split("\n")])
    if '' in tokens:
        tokens.remove('')
    return tokens


def get_stopped_text(text, stop_set):
    """Remove stop words from text after splitting
    on whitespace
    :param text: str
    :param stop_set: set of str
    """
    return " ".join([s for s in TOKENIZER.tokenize(text) if s.lower() not in stop_set])


def get_clean_text(text, keep_set):
    """Downcase and strip punctuation

    :param text: str
    :param keep_set: set of str, words that should be kept
    """
    tmp = [tokenization.clean_punctuation(t).lower() for t in TOKENIZER.tokenize(text)]
    return " ".join([t for t in tmp if t in keep_set])


def get_num_tokens(text):
    """Returns the number of tokens after splitting on whitespace.

    :param text: str
    """
    return len(TOKENIZER.tokenize(text))

def get_retokenized(text):
    return " ".join(TOKENIZER.tokenize(text))


##############################################################################
# Everything that follows is a bunch of data munging that I'm not really
# happy with, but is useful for getting by author statistics.
# I need to think about best way to handle token cleaning for analysis unrelated
# to topic model trainign
##############################################################################


def get_by_author_statistics(corpus_df, author_col, doc_length_col):
    """Returns a dataframe with by author statistics for total number of documents ('doc_count'),
    total number of tokens ('token_count') and normalized document length ('normalized_doc_length')

    :param corpus_df: DataFrame containing the specified columns author_col and doc_length_col where each row is a doc
    :param author_col: str, column containing author ids
    :param doc_length_col: str, column with numeric document length
    """
    author_groupby = corpus_df.groupby(author_col)
    docs_by_author = author_groupby.size().reset_index(name='doc_count')
    tokens_by_author = author_groupby[doc_length_col].sum().reset_index(name='token_count')
    normalized_tokens_by_author = pd.merge(docs_by_author, tokens_by_author, on='author')
    normalized_tokens_by_author['normalized_doc_length'] = normalized_tokens_by_author['token_count'] / normalized_tokens_by_author['doc_count']
    return normalized_tokens_by_author


def get_by_author_word_counts(corpus_df, text_col, author_col):
    """Gets word counts by author using the specified text_col as
    documents. Expects a DataFrame with a column of free text
    where each row is a document and with a categorical column for
    the author. Does counts using sklearn's CountVectorizer with
    the RegexTokenizer.
    # TODO Give CountVectorizer the vocab list from Mallet

    :param corpus_df: The original DataFrame
    :param text_col: The name of the column storing free text
    :param author_col: The name of the column storing author ids
    """
    all_author_counts = pd.DataFrame()
    authors = set(corpus_df[author_col])
    for a in authors:
        author_texts = list(corpus_df[corpus_df[author_col]==a][text_col])
        cv = CountVectorizer(tokenizer = TOKENIZER.tokenize)
        # each column represents a word, each row a document for that author
        author_count = cv.fit_transform(author_texts)
        # sum column to get single word counts for that author over all docs
        total_author_counts = author_count.toarray().sum(axis=0)
        author_df = pd.DataFrame(total_author_counts, columns = [a],
                                 index = cv.get_feature_names())

        # Merge dataframes for each author together.
        all_author_counts = all_author_counts.join(author_df, how='outer')

    # Replace all NaNs with 0 when an author doesn't use a word
    all_author_counts = all_author_counts.fillna(0.0)
    return all_author_counts


def get_by_author_word_counts_all_rows(corpus_df, text_col, author_col):
    """Gets word counts by author using the specified text_col as
    documents. Expects a DataFrame with a column of free text
    where each row is a document and with a categorical column for
    the author. Does counts using Counter and returns
    a DataFrame with columns 'author','token', 'count'

    :param corpus_df: The original DataFrame
    :param text_col: The name of the column storing free text
    :param author_col: The name of the column storing author ids
    """
    results = list()
    authors = set(corpus_df[author_col])
    for a in authors:
        author_texts = list(corpus_df[corpus_df[author_col]==a][text_col])
        author_counter = collections.Counter()
        for doc in author_texts:
            tokenized_doc = TOKENIZER.tokenize(doc)
            author_counter.update(tokenized_doc)

        for token, count in author_counter.items():
            results.append((a, token, count))

    return pd.DataFrame.from_records(results, columns = ['author','token','count'])

def corpus_to_author_token_counts(corpus_path, all_rows=True):
    '''Reads in the original corpus, tokenizes and downcases text (just like
    Mallet would do) and produces by-author token counts.
    You can get results as DataFrame in two ways:
    - all_rows=True: columns are author, token, count
    - all_rows=False: columns are authors, rows are tokens and value at i,j is count

    :param corpus_path: Path to Mallet tsv format corpus with columns doc_id, author, text
    :param all_rows: boolean, select desired DataFrame output format
    '''
    corpus = pd.read_csv(corpus_path, sep='\t', names=['doc_id', 'author', 'text'],
                                    encoding='utf-8')

    if all_rows:
        corpus['clean_text'] = corpus['text'].apply(get_retokenized)
        corpus['clean_text'] = corpus['clean_text'].str.lower()
        return get_by_author_word_counts_all_rows(corpus, 'clean_text', 'author')
    else:
        corpus['clean_text'] = corpus['text'].str.lower()
        return get_by_author_word_counts(corpus, 'clean_text', 'author')


