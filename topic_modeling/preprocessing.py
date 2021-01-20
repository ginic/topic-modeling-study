# coding=utf-8

"""Preprocessing (primarily) Russian raw text for topic modeling.

Script usage: `python preprocessing.py tsv_output_path input_directory`
where tsv_output_path is the Mallet file to write to and input_directory 
contains UTF-8 text files

TODO: this assumes everything is written in UTF-8, but we might have to deal with Windows-1251
"""

__author__= "Virginia Partridge"

import pathlib
import sys

def parse_raw_text_file(file_path):
	"""Raw text file to snippets in list format easily used with Mallet.
	For now, doc_id (doc name) is the label.
	:param file_path: Path a particular text file
	:returns: List of lists like [ [doc_id_0, label, text], 
								   [doc_id_1, label, text] ]
	"""
	counter = 0
	results = []
	doc_id = file_path.stem
	with open(file_path, mode='r', encoding='utf-8') as doc_in:
		snippet = ''
		for line in doc_in:
			if not line.isspace():
				clean_line = line.replace('\t', ' ').strip()
				snippet = ' '.join([snippet, clean_line])

			elif line.isspace() and snippet != '':
				results.append([f'{doc_id}_{counter}', doc_id, snippet])
				counter += 1
				snippet = ''

	return results


def main(tsv_output, input_dir):
	"""Given a directory of input text files, breaks documents into 'paragraphs' and outputs a Mallet format TSV file.
	:param tsv_output: Str, path to desired tsv output file
	:param input_dir: Str, path to directory containing input txt files
	"""
	input_dir_path = pathlib.Path(input_dir)
	for document in input_dir_path.glob('*.txt'):
		print("Parsing doc:", document)
		parsed_doc = parse_raw_text_file(document)
		with open(tsv_output, mode='a', encoding='utf-8') as tsv_out:
			tsv_out.write('\n'.join('\t'.join(l) for l in parsed_doc))


if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])