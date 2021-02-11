# coding=utf-8

"""Preprocessing (primarily) Russian raw text for topic modeling.
Splits documents on blank lines.

Script usage: `python preprocessing.py tsv_output_path input_directory`
where tsv_output_path is the Mallet file to write to and input_directory
contains UTF-8 text files

TODO: Add options for more fine control over splitting documents (split every specified number of tokens or such)
TODO: This assumes everything is written in UTF-8 and skips undecodeable files, but we might have to deal with Windows-1251 too eventually
TODO: Switch from sys.argv to argparse and add help
"""

__author__= "Virginia Partridge"

import pathlib
import sys


def split_doc_on_word_count(word_count=150):
    """Raw text file to snippets of size of word_count word types
    deliniated by white space.
    For now, doc_id (doc name) is the label.

    :param file_path: Path a particular text file
    :returns: List of lists like [ [doc_id_0, label, text],
                                    [doc_id_1, label, text] ]
    :raise: UnicodeDecodeError if file isn't UTF-8 decodeable
    """
    pass

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


def main(tsv_output, input_dir):
    """Given a directory of input text files, breaks documents into 'paragraphs' and outputs a Mallet format TSV file.
    :param tsv_output: Str, path to desired tsv output file
    :param input_dir: Str, path to directory containing input txt files
    """
    input_dir_path = pathlib.Path(input_dir)
    with open(tsv_output, mode='a', encoding='utf-8') as tsv_out:
        for document in input_dir_path.glob('*.txt'):
            print("Parsing doc:", document)
            try:
                parsed_doc = split_doc_on_blank_lines(document)
                tsv_out.writelines(['\t'.join(l) + '\n' for l in parsed_doc])
            except UnicodeDecodeError as e:
                print(document, "doesn't appear to be in UTF-8, skipping. Error:", e)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])