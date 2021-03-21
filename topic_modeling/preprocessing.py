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
"""
import abc
import argparse
import collections
import pathlib
import sys

import topic_modeling.tokenization as tokenization

DEFAULT_WORD_COUNT = 500
DEFAULT_LINE_COUNT = 10
WORD_COUNT_SPLITTER = 'word_count'
LINE_BREAK_SPLITTER = 'line_break'
LINE_COUNT_SPLITTER = 'line_count'


def get_author_label(doc_id):
    """ Return the author from the RussianNovels corpus file name.
    Placeholder until better metadata handling can be put in place.
    """
    return doc_id.split('_')[0]

class DocumentSplitter(abc.ABC):
    def __init__(self, tokenizer=None):
        self.token_counter = collections.Counter()
        if tokenizer is None:
            self.tokenizer = tokenization.RegexTokenizer()

    def split_doc(self, file_path, word_count=DEFAULT_WORD_COUNT,
                  line_count=DEFAULT_LINE_COUNT):
        """ Parses the document to Mallet format in the manner specified by
        split_choice.

        :param file_path: Path a particular text file
        :param split_choice: str, switch 'line_break', 'line_count' or 'word_count'
        :param word_count: int, min number of words allowed in snippet
        :param line_count: int, min number of lines allowed in snippet
        :returns: List of tuples like [ (doc_id_0, label, text),
                                        (doc_id_1, label, text) ]
        :raise: UnicodeDecodeError if file isn't UTF-8 decodeable
        """
        doc_id = file_path.stem
        author_label = get_author_label(doc_id)
        # TODO
        # split_text = self.process_file()


        # TODO Handling metadata here isn't ideal, move when more info on metadata
        num_docs = len(parsed_doc)
        doc_ids = [f'{doc_id}_{i}' for i in range(num_docs)]
        return list(zip(doc_ids, [author_label] * num_docs, parsed_doc))

    @abc.abstractmethod
    def process_file(self, line):
        pass

class LineCountSplitter(DocumentSplitter):

    def __init__(self, tokenizer=None, line_count=DEFAULT_LINE_COUNT,
                 min_word_count=DEFAULT_WORD_COUNT):
        self.line_count = line_count
        self.min_word_count = min_word_count
        super().__init__(tokenizer)


    def process_file(self, file_path):
        """Raw text file broken down into snippets that are at least 'line_count'
        lines long and at least 'min_word_count' word types in length.

        :param file_path: Path a particular text file
        :param line_count: int, min number of lines allowed in snippet
        :param min_word_count: int, min number of words allowed in snippet (except for last snippet)
        :returns: List of tupes like [ [doc_id_0, label, text],
                                        [doc_id_1, label, text] ]
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
            text = ' '.join([' '.join(l) for l in tokenized_lines[start:end]])
            results.append(text)
            start = end
            end = start + self.line_count

        # There could be a final snippet with length < min_word_count, let's keep it
        if tokenized_lines[start:end] != []:
            text = ' '.join([' '.join(l) for l in tokenized_lines[start:]])
            results.append(text)

        return results

class WordCountSplitter(DocumentSplitter):

    def __init__(self, tokenizer=None, word_count=DEFAULT_LINE_COUNT):
        self.word_count=word_count
        super().__init__(tokenizer)

    def process_file(self, file_path):
        """Raw text file broken down to snippets of size of word_count word types
        deliniated by white space.
        For now, doc_id (doc name) is the label.

        :param file_path: Path a particular text file
        :param word_count: int marking the number of words to subdivide documents
        :returns: List of lists like [ [doc_id_0, label, text],
                                        [doc_id_1, label, text] ]
        :raise: UnicodeDecodeError if file isn't UTF-8 decodeable
        """
        results = []
        # Individual documents shouldn't be big enough for this to cause problems
        split_text = self.tokenizer.tokenize(file_path.read_text(encoding='utf-8'))
        for i in range(0, len(split_text), self.word_count):
            results.append(' '.join(split_text[i:i + self.word_count]))

        return results

class LineBreakSplitter(DocumentSplitter):

    def process_file(self, file_path):
        """Raw text file to snippets in list format easily used with Mallet.
        Strips out any tabs from text. Documents are divided on blank lines.
        For now, doc_id (doc name) is the label.

        :param file_path: Path a particular text file
        :returns: List of document snippets [ text1, text2 ]
        :raise: UnicodeDecodeError if file isn't UTF-8 decodeable
        """
        results = []
        doc_id = file_path.stem
        with open(file_path, mode='r', encoding='utf-8') as doc_in:
            snippet = ''
            for line in doc_in:
                if not line.isspace():
                    clean_line = line.replace('\t', ' ').strip()
                    snippet = ' '.join([snippet, clean_line]).strip()
                elif line.isspace() and snippet != '':
                    results.append(snippet)
                    snippet = ''

        # If documents don't end in blank lines, add the last snippet
        if not snippet.isspace():
            results.append(snippet)

        return results


def main(tsv_output, input_dir, split_choice=WORD_COUNT_SPLITTER,
         word_count=DEFAULT_WORD_COUNT, line_count=DEFAULT_LINE_COUNT):
    """Given a directory of input text files, breaks documents into '
    paragraphs' and outputs a Mallet format TSV file.
    :param tsv_output: Str, path to desired tsv output file
    :param input_dir: Str, path to directory containing input txt files
    """
    if split_choice == LINE_BREAK_SPLITTER:
        splitter = LineBreakSplitter()
    elif split_choice == WORD_COUNT_SPLITTER:
        splitter = WordCountSplitter(word_count)
    elif split_choice == LINE_COUNT_SPLITTER:
        splitter = LineCountSplitter(line_count=line_count,
                                     min_word_count=word_count)
    else:
        raise ValueError(f"Unsupported document splitting option:{split_choice}")

    input_dir_path = pathlib.Path(input_dir)
    with open(tsv_output, mode='a', encoding='utf-8') as tsv_out:
        for document in input_dir_path.glob('*.txt'):
            try:
                print("Parsing doc:", document)
                parsed_doc = splitter.split_doc(document)
                tsv_out.writelines(['\t'.join(l) + '\n' for l in parsed_doc])
            except UnicodeDecodeError as e:
                print(document, "doesn't appear to be in UTF-8, skipping. Error:", e)


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


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.tsv_output, args.corpus_directory, args.splitter, args.word_count)