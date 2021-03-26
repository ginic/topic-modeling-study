# coding=utf-8
"""Preprocessing non-English (primarily Russian, but theoretically this
shouldn't matter) raw text for topic modeling.
Options for splitting documents on blank lines or to a limited word count.

Basic script usage: `python preprocessing.py tsv_output_path input_directory`
where tsv_output_path is the Mallet file to write to and input_directory
contains UTF-8 text files

TODO: This assumes everything is written in UTF-8 and skips undecodeable files,
      but we might have to deal with Windows-1251 too eventually
TODO: Properly choose metadata for column 2 given a metadata input file
TODO: Author counts map to its own object for cleanness & usability
"""
import abc
import argparse
import collections
import pathlib
import pickle
import sys

import pandas as pd

import topic_modeling.tokenization as tokenization

DEFAULT_WORD_COUNT = 500
DEFAULT_LINE_COUNT = 10
WORD_COUNT_SPLITTER = 'word_count'
LINE_BREAK_SPLITTER = 'line_break'
LINE_COUNT_SPLITTER = 'line_count'

def unpickle_author_counts(pickle_path):
    '''Returns a unpickled DocumentSplitter.token_counter
    map {author:{doc id:Counter({token:term freq within doc})}}

    :param pickle_path: Path to pickled DocumentSplitter.token_counter
    '''
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def author_counts_as_dataframe(author_counts_map, keep_doc_col=False):
    '''Given an token_counter map from Document splitter, transforms
    into pandas frame with columns: author, token, count.
    There is also an option to keep break down by document id.

    :param author_counts_map: dict, {author:{doc id:Counter({token:term freq within doc})}}
    :param keep_doc_col: boolean, defaults to False. Set to true to keep breakdown by doc_id
    '''
    records = list()
    for a, doc_counts in author_counts_map.items():
        author_count = collections.Counter()
        for _, counts in doc_counts.items():
            author_count = author_count | counts

        for token, count in author_count.items():
            records.append((a, token, count))

    return pd.DataFrame.from_records(records, columns = ['author','token','count'])


def get_author_label(doc_id):
    '''Return the author from the RussianNovels corpus file name.
    Placeholder until better metadata handling can be put in place.
    '''
    return doc_id.split('_')[0]

class DocumentSplitter(abc.ABC):
    '''Abstract base class for splitting documents.
    Attributes are a tokenizer and token_counter which tracks token counts
    by author and file for sanity checking.
    The default tokenization is splitting on whitespace.
    The token_counter maps {author:{doc id:Counter({token:term freq within doc})}}

    Subclasses must implement the process_file method.
    '''
    def __init__(self, tokenizer=None):
        self.token_counter = collections.defaultdict(dict)
        if tokenizer is None:
            self.tokenizer = tokenization.RegexTokenizer(
                tokenization.NON_WHITESPACE_TOKENIZATION)

    def split_doc(self, file_path):
        """Parses the document to Mallet format in the manner specified by
        split_choice.

        :param file_path: Path a particular text file
        :param split_choice: str, switch 'line_break', 'line_count' or 'word_count'
        :param word_count: int, min number of words allowed in snippet
        :param line_count: int, min number of lines allowed in snippet
        :returns: List of tuples like [ (doc_id_0, label, text),
                                        (doc_id_1, label, text) ]
        :raise: UnicodeDecodeError if file isn't UTF-8 decodeable
        """
        # TODO Handling metadata here isn't ideal, move when more info on metdata
        doc_id = file_path.stem
        author_label = get_author_label(doc_id)
        # Count up tokens for this document
        # word type -> count for this file
        counter = collections.Counter()
        parsed_doc = self.process_file(file_path)
        for d in parsed_doc:
            counter.update(d)
        self.token_counter[author_label][doc_id] = counter

        num_docs = len(parsed_doc)
        doc_ids = [f'{doc_id}_{i}' for i in range(num_docs)]
        joined_docs = [' '.join(l) for l in parsed_doc]
        return list(zip(doc_ids, [author_label] * num_docs, joined_docs))

    @abc.abstractmethod
    def process_file(self, file_path):
        '''Implementation to process the individual file into smaller documents and update `token_counter` during the process
        :returns: A list of lists of tokens
            [[doc1_token1, doc1_token2,...], [doc2_token1, doc2_token2...]]
        '''
        pass

    def print_authorial_statistics(self):
        '''Prints the document, vocab and token count statistics for each author
        collected by this DocSplitter for each input file.
        '''
        overall_tokens = 0
        overall_vocab = set()
        file_count = 0
        for a in self.token_counter.keys():
            author_tokens = 0
            author_vocab = set()
            author_file_count = len(self.token_counter[a])
            file_count += author_file_count
            print("Authorial stats for:", a)
            print(f"\t{author_file_count} file(s) processed")
            for f in self.token_counter[a]:
                file_counter = self.token_counter[a][f]
                vocab = set(file_counter)
                num_tokens = sum(file_counter.values())
                print(f"\tFile:'{f}'\tVocab size: {len(vocab)}\tToken counts: {num_tokens}")
                author_tokens += num_tokens
                author_vocab.update(vocab)
            print(f"\tAuthor: '{a}'\tTotal vocab size: {len(author_vocab)}\tTotal tokens: {author_tokens}")
            overall_tokens += author_tokens
            overall_vocab.update(author_vocab)
        print("---------------------------------------")
        print("Summary over all authors:")
        print("Total files processed:", file_count)
        print("Vocab size:", len(overall_vocab))
        print("Token count:", overall_tokens)


class LineCountSplitter(DocumentSplitter):
    '''Raw text file broken down into snippets that are at least `line_count` lines long and at least `min_word_count`
    word types in length.

    Attributes inherited from DocumentSplitter are a tokenizer and token_counter.
    '''
    def __init__(self, tokenizer=None, line_count=DEFAULT_LINE_COUNT,
                 min_word_count=DEFAULT_WORD_COUNT):
        self.line_count = line_count
        self.min_word_count = min_word_count
        super().__init__(tokenizer)


    def process_file(self, file_path):
        """Raw text file broken down into snippets that are at least `line_count`
        lines long and at least `min_word_count` word types in length.

        :param file_path: Path a particular text file
        :returns: List of lists of tokens
        :raise: UnicodeDecodeError if file isn't UTF-8 decodeable
        """
        results = []
        start = 0
        # Individual documents shouldn't be big enough for this to cause problems
        lines = file_path.read_text(encoding='utf-8').split('\n')
        # Empty lines are excluded from line count
        # [[l_1token1, l1_token2], [l2_token1, l2_token2]]
        tokenized_lines = [self.tokenizer.tokenize(l) for l in lines if not l.isspace() and l != '']
        num_lines = len(tokenized_lines)
        end = self.line_count

        while end < num_lines:
            token_count = sum(map(len, tokenized_lines[start:end]))
            # Get to at least min_word_count_tokens
            while token_count < self.min_word_count and end < num_lines:
                end += 1
                token_count += len(tokenized_lines[end - 1])
            results.append([t for l in tokenized_lines[start:end] for t in l])
            start = end
            end = start + self.line_count

        # There could be a final snippet with length < min_word_count, let's keep it
        if tokenized_lines[start:end] != []:
            results.append([t for l in tokenized_lines[start:] for t in l])

        return results

class WordCountSplitter(DocumentSplitter):
    '''Breaks raw text file down into snippets that are the length of the
    specified word_count.
    Attributes inherited from DocumentSplitter are a tokenizer and token_counter.
    '''
    def __init__(self, tokenizer=None, word_count=DEFAULT_LINE_COUNT):
        self.word_count=word_count
        super().__init__(tokenizer)

    def process_file(self, file_path):
        """Raw text file broken down to snippets of size of word_count.

        :param file_path: Path a particular text file
        :returns: List of lists of tokens
        :raise: UnicodeDecodeError if file isn't UTF-8 decodeable
        """
        results = []
        # Individual documents shouldn't be big enough for this to cause problems
        split_text = self.tokenizer.tokenize(file_path.read_text(encoding='utf-8'))
        for i in range(0, len(split_text), self.word_count):
            results.append(split_text[i:i + self.word_count])

        return results

class LineBreakSplitter(DocumentSplitter):

    def process_file(self, file_path):
        """Raw text file to snippets in list format easily used with Mallet.
        Strips out any tabs from text. Documents are divided on blank lines.
        For now, doc_id (doc name) is the label.

        :param file_path: Path a particular text file
        :returns: List of lists of tokens
        :raise: UnicodeDecodeError if file isn't UTF-8 decodeable
        """
        results = []
        with open(file_path, mode='r', encoding='utf-8') as doc_in:
            snippet = ''
            for line in doc_in:
                if not line.isspace():
                    clean_line = line.replace('\t', ' ').strip()
                    snippet = ' '.join([snippet, clean_line]).strip()
                elif line.isspace() and snippet != '':
                    results.append(self.tokenizer.tokenize(snippet))
                    snippet = ''

        # If documents don't end in blank lines, add the last snippet
        if not snippet.isspace():
            results.append(self.tokenizer.tokenize(snippet))

        return results


def main(tsv_output, input_dir, split_choice=WORD_COUNT_SPLITTER,
         word_count=DEFAULT_WORD_COUNT, line_count=DEFAULT_LINE_COUNT,
         pickle_counts_path=None):
    """Given a directory of input text files, breaks documents into '
    paragraphs' and outputs a Mallet format TSV file.
    :param tsv_output: str, path to desired tsv output file
    :param input_dir: str, path to directory containing input txt files
    :param split_choice: str, option for chooseing document split method
    :param word_count: int, exact word count for WordCountSlitter and minimum_word_count for LineCountSplitter
    :param line_count: int, minimum line count for LineCountSplitter
    :param pickle_counts_path: str, optional path to a file for pickling the token count mapping
    """
    print("Line splitter choice:", split_choice)
    if split_choice == LINE_BREAK_SPLITTER:
        splitter = LineBreakSplitter()
    elif split_choice == WORD_COUNT_SPLITTER:
        splitter = WordCountSplitter(word_count=word_count)
        print("Word (token) count for split documents:", word_count)
    elif split_choice == LINE_COUNT_SPLITTER:
        splitter = LineCountSplitter(line_count=line_count,
                                     min_word_count=word_count)
        print("Minimum line count for split documents:", line_count)
        print("Minimum word (token) count for split documents:", word_count)
    else:
        raise ValueError(f"Unsupported document splitting option:{split_choice}")

    input_dir_path = pathlib.Path(input_dir)
    with open(tsv_output, mode='a', encoding='utf-8') as tsv_out:
        for document in input_dir_path.glob('*.txt'):
            try:
                print("Parsing file:", document)
                parsed_doc = splitter.split_doc(document)
                print("File", document, "split into", len(parsed_doc), "documents")
                tsv_out.writelines(['\t'.join(l) + '\n' for l in parsed_doc])
            except UnicodeDecodeError as e:
                print(document, "doesn't appear to be in UTF-8, skipping. Error:", e)

    splitter.print_authorial_statistics()

    if pickle_counts_path:
        with open(pickle_counts_path, 'wb') as f:
            pickle.dump(splitter.token_counter, f)


parser = argparse.ArgumentParser(description="Preprocessing of text for topic modeling with Mallet")
parser.add_argument(
    'tsv_output',
    help="Path to desired TSV output files. Documents will be appended to the document if it already exists.",)
parser.add_argument(
    'corpus_directory',
    help="Path to the folder containing corpus where each document is expected to be a .txt file")
parser.add_argument(
    '--splitter',
    help='Choice of heuristic for splitting documents. Default "word_count" splits into specified number of tokens. The "line_count" option splits documents into at least the specified number of lines, but will add lines until the word count minimum is reached. The "line_break" option splits at blank lines.',
    default=LINE_COUNT_SPLITTER,
    choices=[LINE_BREAK_SPLITTER, WORD_COUNT_SPLITTER, LINE_COUNT_SPLITTER])
parser.add_argument(
    '--word_count', '-w',
    type=int,
    default=DEFAULT_WORD_COUNT,
    help="Document size in word type count when splitting using 'word_count' option. Minimum number of words allowed in a document when using the 'line_count' option.")
parser.add_argument(
    '--line_count', '-l',
    type=int,
    default=DEFAULT_LINE_COUNT,
    help="Minimum number of lines "
)
parser.add_argument(
    '--pickle-counts', '-p',
    help="Optional path to pickle token counts by author and file"
)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.tsv_output, args.corpus_directory, args.splitter, args.word_count, args.line_count, args.pickle_counts)