# coding=utf-8
"""Preprocessing operations for specially formatted corpora, such as OpenCorpora (http://opencorpora.org), Russian National Corpus (https://ruscorpora.ru/old/en/index.html) or TIGER (https://www.ims.uni-stuttgart.de/en/research/resources/corpora/tiger/).

Assumes you do not want to subdivide documents (unlike in the preprocessing.py
for raw text documents), just write each corpus document to a single line in the TSV document. Appropriate stopword removal and stemming is also performed.

Input options:
- Choose the corpus format:
    - OpenCorpora: Expect the entire corpus in a single XML file (UTF-8 encoding) (http://opencorpora.org/files/export/annot/annot.opcorpora.xml.zip)
    - Russian National Corpus: Folders (per genre) of many XML files (Windows-1251 encoding)
    - TIGER: Expect the entire corpus as a single XML file (for description of format https://www.ims.uni-stuttgart.de/documents/ressourcen/werkzeuge/tigersearch/doc/html/TigerXML.html)
- Choose a genre restriction. There are different options depending on the corpus:
    - OpenCorpora: newspaper, encyclopedia, blogs, literary, nonfiction, legal
    - RNC: blogs, fiction, public, science, speech
Output options:
 - Output directory: To store processed MALLET TSV files for each stemmer and the mophology oracle TSV when this is available.
"""
# TODO separate corpus iterator from stemmer writing functionality
from abc import ABC, abstractmethod
import argparse
from collections import defaultdict
import gzip
from pathlib import Path
import regex
import xml.etree.ElementTree as ET

from spacy.lang.ru import Russian
from spacy.lang.de import German

import topic_modeling.stemming as stemming
from topic_modeling.tokenization import WORD_TYPE_NO_DIGITS_TOKENIZATION

# Corpus name constants
RNC = "rnc"
OPENCORPORA = "opencorpora"
TIGER = "tiger"


RAW = "raw" # Use for unstemmed surface forms
ORACLE = "oracle" # Use for oracle lemmatized forms

class TSVWriter:
    """A simple TSV writer wrapper"""
    def __init__(self, tsv_path, as_gzip=False):
        """
        :param tsv_path: The path to the TSV file
        :param as_gzip: Set to True to write TSV to a compressed GZIP
        """
        self.tsv_path = tsv_path
        self.tsv_writer = None
        self.as_gzip = as_gzip

    def open(self):
        if self.as_gzip:
            self.tsv_writer = gzip.open(self.tsv_path, mode='wt', encoding='utf-8')
        else:
            self.tsv_writer = open(self.tsv_path, mode='w', encoding='utf-8')

    def write_row(self, list_of_elements, delimiter='\t'):
        self.tsv_writer.write(delimiter.join(list_of_elements) + '\n')

    def close(self):
        self.tsv_writer.close()


class TSVDocumentWriter(TSVWriter):
    """Defines writing out to Mallet TSV format.
    Also tracks number of documents written.
    """

    def __init__(self, tsv_path):
        self.num_docs = 0
        super().__init__(tsv_path)

    def write_doc(self, doc_id, metadata, text):
        self.write_row([doc_id, metadata, text])
        self.num_docs += 1
        if self.num_docs % 50 == 0:
            print(f"Wrote doc {doc_id} to {self.tsv_path}. {self.num_docs} doc(s) written.")


class TSVOracleWriter(TSVWriter):
    """Defines writing out the morphology oracle to a gzipped TSV to save space.
    This writes out the gold standard lemma and morphological analysis
    assigned for each surface form in the corpus.
    The format is similar to Mallet state files:
    corpus_document_id\tdocument_index\tsurface_form\toracle_lemma\tmorphological_analysis
    """

    def __init__(self, tsv_path):
        self.current_doc_id = None
        # This will be aligned with Mallet's state file document ids
        self.doc_index = -1
        super().__init__(tsv_path, as_gzip=True)

    def write_entry(self, doc_id, surface_form, oracle_lemma, morph_analysis):
        """Writes an entry the corpus mophology oracle TSV.

        :param doc_id: str, the document id used by the original corpus
        :param surface_form: str, the surface realization of the word type
        :param oracle_lemma: str, the lemma assigned in corpus gold standard
        :param morph_analysis: str, the morphological analysis assigned in corpus gold standard
        """
        if not doc_id == self.current_doc_id:
            self.current_doc_id = doc_id
            self.doc_index += 1

        self.write_row([self.current_doc_id, str(self.doc_index), surface_form,
                        oracle_lemma, morph_analysis])


class CorpusParser(ABC):
    """What behaviours do we want corpus parsers to have?
        - filter using stop word list, using stopwords from spacy
        - run each doc through lemmatizer/stemmer options
        - check that the lengths of each document are the same for each stemmer
    """

    def __init__(self, language, corpus_name, corpus_path, has_oracle=False):
        """
        :param language: str, two letter language code corresponding to spaCy
        :param corpus_name: str, identifying for the corpus, used to name TSV output files
        :param corpus_path: str or Path, path to the root directory or file(s) relevant to the corpus
        :param has_oracle: True if the corpus has gold standard annotations for lemmas and morphological analysis

        """
        self.language = language
        self.corpus_name = corpus_name
        self.corpus_path = Path(corpus_path)
        self.has_oracle = has_oracle


    @abstractmethod
    def document_generator(self):
        """A generator function to iterate over documents in the corpus.
        Should yield document id and the entry point to access the
        document (e.g. XML root or file path)
        """
        pass

    @abstractmethod
    def token_generator(self, document_root):
        """A generator function to iterate over all tokens in the given document
        """
        pass


class RussianNationalCorpusParser(CorpusParser):
    """
    """
    def __init__(self, rnc_root, filters=['public']):
        """
        :param rnc_root: str or Path, The root path of the RNC, a folder containing subfolders 'TABLES' and 'TEXT'
        :param filters: list of str, The names of subcorpus folders you'd like to process
        """
        self.filters = filters
        super().__init__("ru", RNC, rnc_root, has_oracle=True)

    def document_generator(self):
        """Generator function to iterate over all the documents in the corpus.
        Returns the path to a single XML file
        """
        print("Starting iteration of the Russian National Corpus")
        for subcorpus in Path(self.corpus_path / 'TEXTS').iterdir():
            if subcorpus.name in self.filters and subcorpus.is_dir():
                print("Parsing subcorpus:", subcorpus.name)
                for xml_file in subcorpus.iterdir():
                    doc_id = xml_file.stem
                    yield doc_id, xml_file
            else:
                print("Skipping subcorpus:", subcorpus.name)

    def token_generator(self, xml_path):
        """Generator function iterate over all tokens in the XML file.
        Return the surface form, the oracle form and the morphological analysis. If there is no oracle form or morphological analysis, these will be None.
        """
        tree =  ET.parse(xml_path)
        for word_root in tree.iter(tag='w'):
            # According to RNC guidelines, multiple morphological analyses are possible
            # in some ambiguous cases. Let's catch them all.
            lemmas = []
            grs = []
            try:
                for ana_root in word_root.iter(tag='ana'):
                    # Surface form will be after the last annotation tag for a word
                    annotation_tail = ana_root.tail
                    lemmas.append(ana_root.get('lex'))
                    grs.append(ana_root.get('gr'))

                # Strip out stress markers from surface form
                surface_form = annotation_tail.replace('`', '')

                yield surface_form, "|".join(lemmas), "|".join(grs)
            except AttributeError as e:
                print(f"ERROR: Problem at tag with annotation tail '{annotation_tail}', lex '{lemmas}', gr '{grs}'")
                raise e


class OpenCorporaParser(CorpusParser):
    """
    """
    def document_generator(self):
        # TODO
        pass

    def token_generator(self, document_root):
        # TODO
        pass


class TIGERCorpusParser(CorpusParser):
    """
    """
    def document_generator(self):
        # TODO
        pass

    def token_generator(self, document_root):
        # TODO
        pass

class CorpusPreprocessor:
    def __init__(self, corpus_parser, output_dir, use_oracle=True, use_truncation_stemmers = True, truncation_settings = None):
        """
        :param corpus_parser: A CorpusParser object
        :param output_dir: The output directory to write all TSV files to
        :param use_oracle: True to also output a Mallet TSV file with the oracle lemmas as text
        :param use_truncation_stemmers: True to also use the naive stememrs that truncate to certain number of characters
        :param truncation_settings: A list of ints to truncate to if truncation stemmers are used
        """
        self.corpus_parser = corpus_parser
        # Ensure output dir exists
        self.output_dir_path = Path(output_dir)
        self.output_dir_path.mkdir(parents=True)
        self.use_oracle = use_oracle & self.corpus_parser.has_oracle

        # This will be used to throw out any tokens containing digits or only punctuation
        self.word_type_pattern = regex.compile(WORD_TYPE_NO_DIGITS_TOKENIZATION)

        # Defaults for truncation settings
        self.use_trunaction = use_truncation_stemmers
        if self.use_trunaction and truncation_settings is None:
            self.truncation_settings = [5, 6]
        else:
            self.truncation_settings = truncation_settings

        self.set_stopwords()

        # All output writers get configured along with stemmers
        # Will be a TSVOracleWriter if desired
        self.oracle_tsv = None
        # Stores output tsvs for each stemmer type
        self.output_tsvs = {}
        self.set_stemmers()

    def set_stopwords(self):
        """Set stopwords for this corpus based on language.
        """
        # Target language directly for stopword lists
        if self.corpus_parser.language=='ru':
            self.stopwords = Russian().Defaults.stop_words
        elif self.corpus_parser.language=='de':
            self.stopwords = German().Defaults.stop_words
        print(f"Loaded {len(self.stopwords)} stopwords from spaCy for language '{self.corpus_parser.language}'")

    def get_experiment_basename(self, stemmer_name):
        return "_".join([self.corpus_parser.corpus_name, stemmer_name])

    def get_oracle_gzip_name(self):
        return "_".join([self.corpus_parser.corpus_name, "oracleAnalysis"]) + ".gz"

    def set_stemmers(self):
        """Set up stemmers/lemmatizers for this corpus based on language,
        corpus characteristics and user settings for truncation.
        Also configures output filenames for each stemmer type.
        """
        self.stemmers = {}
        self.stemmers.update(stemming.get_language_specific_stemmers(self.corpus_parser.language))

        if self.use_trunaction:
            for i in self.truncation_settings:
                self.stemmers[f'truncate{i}'] = stemming.TruncationStemmer(num_chars=i)

        self.output_tsvs = {}
        for s in self.stemmers:
            experiment_id = self.get_experiment_basename(s)
            experiment_dir = self.output_dir_path / experiment_id
            experiment_dir.mkdir()
            self.output_tsvs[s] = TSVDocumentWriter(experiment_dir / Path(experiment_id + '.tsv'))

        # An oracle lemmatizer is available
        if self.use_oracle:
            oracle_experiment_id = self.get_experiment_basename(ORACLE)
            oracle_dir = self.output_dir_path / oracle_experiment_id
            oracle_dir.mkdir()
            self.output_tsvs[ORACLE] = TSVDocumentWriter(oracle_dir / Path(oracle_experiment_id + '.tsv'))
            self.oracle_tsv = TSVOracleWriter(oracle_dir / Path(self.get_oracle_gzip_name()))

        # Raw text folder and TSV
        raw_experiment_id = self.get_experiment_basename(RAW)
        raw_dir = self.output_dir_path / raw_experiment_id
        raw_dir.mkdir()
        self.output_tsvs[RAW] = TSVDocumentWriter(raw_dir / Path(raw_experiment_id + '.tsv'))


    def open_all(self):
        """Open all TSV writers"""
        for _, writer in self.output_tsvs.items():
            writer.open()

        if self.use_oracle:
            self.oracle_tsv.open()

    def close_all(self):
        """Close all TSV writers"""
        for _, writer in self.output_tsvs.items():
            writer.close()

        if self.use_oracle:
            self.oracle_tsv.close()

    def parse_corpus(self):
        """Iterate through all documents in the corpus. Perform stopword removal
        and stemming, writing out a Mallet TSV for each stemmer/lemmatizer option.
        """
        self.open_all()
        for doc_id, doc_root in self.corpus_parser.document_generator():
            try:
                self.process_document(doc_id, doc_root)
            except Exception as e:
                print("Error processing document with doc_id:", doc_id)
                raise e
        self.close_all()
        self.validate_document_counts()
        print(f"{self.output_tsvs[RAW].num_docs} documents processed")
        print(f"Finished processing corpus {self.corpus_parser.corpus_name}")

    def use_token(self, token):
        """Return True if token isn't a stopword, has digits or is only punctuation
        :param token: str, an unstemmed surface form
        """
        if token in self.stopwords:
            return False
        if self.word_type_pattern.fullmatch(token) is None:
            return False
        return True

    def validate_document_counts(self):
        """Raises ValueError if document writer counts are different.
        """
        doc_count_mapping = set([v.num_docs for v in self.output_tsvs.values()])
        doc_count_mapping.add(self.oracle_tsv.doc_index + 1)
        if len(doc_count_mapping) > 1:
            message = "\n".join([f"{k}:{v.num_docs} docs" for k,v in self.output_tsvs.items()])
            message += f"\nOracle Gzip: {self.oracle_tsv.doc_index + 1}  docs"
            raise ValueError("TSV Writers wrote different numbers of documents!\n" + message)


    def process_document(self, doc_id, doc_root):
        """
        :param doc_id: str, document identifier used by the corpus
        :param doc_root: Root entry point for a document in the corpus. Could be pretty much anything, just pass back to corpus parser
        """
        # Track the document tokens by stemmer type
        processed_docs_map = defaultdict(list)
        for token, oracle, morph_analysis in self.corpus_parser.token_generator(doc_root):
            if self.use_token(token):
                processed_docs_map[RAW].append(token)
                if self.use_oracle:
                    processed_docs_map[ORACLE].append(oracle)
                    self.oracle_tsv.write_entry(doc_id, token, oracle, morph_analysis)
                for stemmer_name, stemmer in self.stemmers.items():
                    lemma = stemmer.single_term_lemma(token).strip()
                    # Don't keep empty strings, fall back to the original token
                    if lemma=="":
                        processed_docs_map[stemmer_name].append(token)
                    else:
                        processed_docs_map[stemmer_name].append(lemma)

        # Validate that all stemmers produce the same number of word forms
        token_lengths = set(map(len, processed_docs_map.values()))
        if len(token_lengths) > 1:
            gather_results = "\n".join([f"STEMMER: {k} LENGTH: {len(v)} TOKENS: {v}" for k,v in processed_docs_map.items()])
            raise ValueError(f"Some stemmer produced a different length output for doc id {doc_id}:\n" + gather_results)

        # If everything went well, write the outputs to TSV
        for stemmer_name, tokens in processed_docs_map.items():
            self.output_tsvs[stemmer_name].write_doc(doc_id, "metadata placeholder", " ".join(tokens))



def get_corpus_parser(corpus_name, corpus_path, **kwargs):
    """Returns the appropriate CorpusParser object for the corpus.

    :param corpus_path: str or Path, path to the root directory or file(s) relevant to the corpus
    :param output_dir: The output directory to write all TSV files to
    """
    if corpus_name==RNC:
        return RussianNationalCorpusParser(corpus_path, **kwargs)
    elif corpus_name==OPENCORPORA:
        return OpenCorporaParser(corpus_path, **kwargs)
    elif corpus_name==TIGER:
        return TIGERCorpusParser(corpus_path, **kwargs)
    else:
        raise ValueError(f"Invalid corpus choice: {corpus_name}")


def main(corpus_name, corpus_in, output_dir):
    corpus_parser = get_corpus_parser(corpus_name, corpus_in)
    preprocessor = CorpusPreprocessor(corpus_parser, output_dir)
    preprocessor.parse_corpus()


parser = argparse.ArgumentParser(description = "Preprocessing for corpora with "\
    "very specific structure and format, including stopword removal, language "\
    "appropriate stemming. Writes out preprocessed corpora to Mallet's TSV "\
    "format and also writes out a TSV with oracular morphological information "\
    "for terms when this is available. SpaCy's stop word list is used.")
parser.add_argument("--corpus_in", nargs=1, required=True,
    help="Path to corpus input file. Could be a single file or single "\
        "directory depending on the corpus.")
parser.add_argument("--corpus_name", type=str, required=True,
    choices=[RNC, OPENCORPORA, TIGER],
    help="Identifier for the corpus." )
parser.add_argument("--output_dir", required=True,
    help="Path to output directory for all Mallet TSV files.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.corpus_name, args.corpus_in[0], args.output_dir)