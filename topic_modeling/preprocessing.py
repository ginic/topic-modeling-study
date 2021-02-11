# coding=utf-8

"""Preprocessing non-English (primarily Russian, but theoretically this shouldn't matter)
raw text for topic modeling. Options for splitting documents on blank lines or to a limited word count.

Basic script usage: `python preprocessing.py tsv_output_path input_directory`
where tsv_output_path is the Mallet file to write to and input_directory
contains UTF-8 text files

TODO: This assumes everything is written in UTF-8 and skips undecodeable files, but we might have to deal with Windows-1251 too eventually
"""

__author__= "Virginia Partridge"

import argparse
import pathlib
import sys

DEFAULT_WORD_COUNT = 150
WORD_COUNT_SPLITTER = 'word_count'
LINE_SPLITTER = 'line'


def split_doc_on_word_count(file_path, word_count=DEFAULT_WORD_COUNT):
    """Raw text file broken down to snippets of size of word_count word types
    deliniated by white space.
    For now, doc_id (doc name) is the label.

    :param file_path: Path a particular text file
    :param word_count: int marking the number of words to subdivide documents in to
    :returns: List of lists like [ [doc_id_0, label, text],
                                    [doc_id_1, label, text] ]
    :raise: UnicodeDecodeError if file isn't UTF-8 decodeable
    """
    results = []
    counter = 0
    doc_id = file_path.stem

    # Individual documents shouldn't be big enough for this to cause problems
    split_text = file_path.read_text(encoding='utf-8').split()
    for i in range(0, len(split_text), word_count):
        snippet = split_text[i:i+word_count]
        results.append([f'{doc_id}_{counter}', doc_id, ' '.join(snippet)])
        counter += 1

    return results


def split_doc_on_blank_lines(file_path):
    """Raw text file to snippets in list format easily used with Mallet. Strips out any tabs from text. Documents are divided on blank lines.
    For now, doc_id (doc name) is the label.

    :param file_path: Path a particular text file
    :returns: List of lists like [ [doc_id_0, label, text],
                                    [doc_id_1, label, text] ]
    :raise: UnicodeDecodeError if file isn't UTF-8 decodeable
    """
    counter = 0
    results = []
    doc_id = file_path.stem
    with open(file_path, mode='r', encoding='utf-8') as doc_in:
        snippet = ''
        for line in doc_in:
            if not line.isspace():
                clean_line = line.replace('\t', ' ').strip()
                snippet = ' '.join([snippet, clean_line]).strip()

            elif line.isspace() and snippet != '':
                results.append([f'{doc_id}_{counter}', doc_id, snippet])
                counter += 1
                snippet = ''

    # If documents don't end in blank lines, add the last snippet
    if not snippet.isspace():
        results.append([f'{doc_id}_{counter}', doc_id, snippet])

    return results


def main(tsv_output, input_dir, split_choice=WORD_COUNT_SPLITTER, word_count=DEFAULT_WORD_COUNT):
    """Given a directory of input text files, breaks documents into 'paragraphs' and outputs a Mallet format TSV file.
    :param tsv_output: Str, path to desired tsv output file
    :param input_dir: Str, path to directory containing input txt files
    """
    input_dir_path = pathlib.Path(input_dir)
    with open(tsv_output, mode='a', encoding='utf-8') as tsv_out:
        for document in input_dir_path.glob('*.txt'):
            print("Parsing doc:", document)
            try:
                if split_choice == LINE_SPLITTER:
                    parsed_doc = split_doc_on_blank_lines(document)
                elif split_choice == WORD_COUNT_SPLITTER:
                    parsed_doc = split_doc_on_word_count(document, word_count)
                else:
                    raise ValueError(f"Unsupported document splitting option:{split_choice}")

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
    help='Choice of heuristic for splitting documents. Default "word_count" splits into a specified number of whitespace deliniated word types and "line" splits at blank lines.',
    default=WORD_COUNT_SPLITTER,
    choices=[LINE_SPLITTER, WORD_COUNT_SPLITTER])
parser.add_argument(
    '--word_count', '-w',
    type=int,
    default=DEFAULT_WORD_COUNT,
    help="Document size in word type count when splitting using 'word_count' option")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.tsv_output, args.corpus_directory, args.splitter, args.word_count)