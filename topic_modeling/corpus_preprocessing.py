# coding=utf-8
"""Preprocessing operations for specially formatted corpora, such as OpenCorpora (http://opencorpora.org), Russian National Corpus (https://ruscorpora.ru/old/en/index.html) or TIGER (https://www.ims.uni-stuttgart.de/en/research/resources/corpora/tiger/).

Assumes you do not want to subdivide documents (unlike in the preprocessing.py
for raw text documents), just write each corpus document to a single line in the TSV document.

Input options:
- Choose the corpus format:
    - OpenCorpora: Expect the entire corpus in a single XML file (UTF-8 encoding) (http://opencorpora.org/files/export/annot/annot.opcorpora.xml.zip)
    - Russian National Corpus: Folders (per genre) of many XML files (Windows-1251 encoding)
    - TIGER: Expect the entire corpus as a single XML file (for description of format https://www.ims.uni-stuttgart.de/documents/ressourcen/werkzeuge/tigersearch/doc/html/TigerXML.html)
- Choose a genre restriction. There are different options depending on the corpus:
    - OpenCorpora: newspaper, encyclopedia, blogs, literary, nonfiction, legal
    - RNC: blogs, fiction, public, science, speech
Output options:
 - TSV file: Process directly to the MALLET TSV format. First column is document ID, second column is the publication/whatever metadata info, third column is the text
"""
from abc import ABC, abstractmethod
import argparse

class TSVDocumentWriter:
    """Defines writing out to Mallet TSV format.
    Also tracks number of documents written.
    """

    def __init__(self, tsv_path):
        self.num_docs = 0
        self.tsv_path = tsv_path
        self.tsv_writer = None

    def open(self):
        self.tsv_writer = open(self.tsv_path, mode='a', encoding='utf-8')

    def write_doc(self, doc_id, metadata, text):
        self.tsv_writer.write(['\t'.join([doc_id, metadata, text])] + '\n')
        self.num_docs += 1
        print(f"Wrote doc {doc_id} to {self.tsv_path}. {self.num_docs} doc(s) written.")

    def close(self):
        self.tsv_writer.close()

# What behaviours do we want corpus parsers to have?
# - filter using stop word list, using stopwords from spacy
# - run each doc through lemmatizer/stemmer options
# - check that the lengths of each document are the same for each stemmer
class CorpusParser(ABC):
    def __init__(self, language):
        self.language = language
        self.set_stopwords()


    @abstractmethod
    def has_next_doc(self):
        pass

    @abstractmethod
    def get_next_doc(self):
        pass




class RussianNationalCorpusParser(CorpusParser):
    """
    """

