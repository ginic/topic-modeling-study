# coding=utf-8
"""Preprocessing operations for specially formatted corpora, such as OpenCorpora (http://opencorpora.org),
Russian National Corpus (https://ruscorpora.ru/old/en/index.html) or
TIGER (https://www.ims.uni-stuttgart.de/en/research/resources/corpora/tiger/).

Assumes you do not want to subdivide documents (unlike in preprocessing.py
for raw text documents), just write each corpus document to a single line in the TSV document.
Appropriate stopword removal and stemming is also performed.

Input options:
- Choose the corpus format:
    - OpenCorpora: Expect the entire corpus in a single XML file (UTF-8 encoding)
      (http://opencorpora.org/files/export/annot/annot.opcorpora.xml.zip)
    - Russian National Corpus: Folders (per genre) of many XML files (Windows-1251 encoding)
    - TIGER: Expect the entire corpus as a single XML file and also requires the 'documents.tsv' file from  TIGER 2.2.doc (additional download).
      (for download and detailed description, see https://www.ims.uni-stuttgart.de/documents/ressourcen/werkzeuge/tigersearch/doc/html/TigerXML.html)
- Choose a genre restriction. There are different options depending on the corpus:
    - OpenCorpora: newspaper, encyclopedia, blogs, literary, nonfiction, legal
    - RNC: blogs, fiction, public, science, speech
Output options:
 - Output directory: To store processed MALLET TSV files for each stemmer and the mophology oracle TSV when this is available.
"""
from abc import ABC, abstractmethod
import argparse
from collections import defaultdict
import csv
import gzip
from pathlib import Path
import xml.etree.ElementTree as ET

import regex
from spacy.lang.ru import Russian
from spacy.lang.de import German

import topic_modeling.stemming as stemming
from topic_modeling.tokenization import WORD_TYPE_NO_DIGITS_TOKENIZATION

# Corpus name constants
RNC = "rnc"
OPENCORPORA = "opencorpora"
TIGER = "tiger"

SLOT_SEP = "," # Use to separate different grammatical slot info in morphological analyisis
AMBIGUOUS_ANALYSIS_SEP = "|" # Use to separate multiple lemmas or analyses when ambiguous


RAW = "raw" # Use for unstemmed surface forms
ORACLE = "oracle" # Use for oracle lemmatized forms

# XML event flags
START = "start"
END = "end"


def advance_xml_iterator(xml_iter, tag, event_flag, document_root_tag):
    """Advances an xml iterator to the specified tag without going past the document root.
    :param xml_iter: ETree iterparse generator
    :param tag: str, desired tag
    :param event_flag: desired "end" or "start" event
    :param document_root_tag: str, the tag that forms the root of the xml document
    """
    done = False
    while not done:
        event, elem = next(xml_iter)
        if event == event_flag and elem.tag == tag:
            done = True
        # Avoid StopIteration exception
        elif event == END and elem.tag == document_root_tag:
            done = True
        else:
            elem.clear()


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
        """Writes a line of data to the tsv file.
        :param list_of_elements: list of str, each element will become a column
        :param delimiter: str, separator for the file, defaults to '\t' for tab separated
        """
        self.tsv_writer.write(delimiter.join(list_of_elements) + '\n')

    def close(self):
        self.tsv_writer.close()


class TSVDocumentWriter(TSVWriter):
    """Defines writing out to Mallet TSV format.
    Also tracks number of documents written.
    Mallet TSV format:
    doc_id\tmetadata\tdocument text
    """

    def __init__(self, tsv_path):
        self.num_docs = 0
        super().__init__(tsv_path)

    def write_doc(self, doc_id, metadata, text):
        """Writes the document to the Mallet TSV file
        :param doc_id: str, document identifier
        :param metadata: str, metadata to store in the second column, not used for anything at the moment
        :param text: str, the document text
        """
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
    """Abstract class to define required methods for iterating over a corpus in the topic modeling context.
    """
    def __init__(self, language, corpus_name, corpus_path, has_oracle=False, filters=None):
        """
        :param language: str, two letter language code corresponding to spaCy
        :param corpus_name: str, identifying for the corpus, used to name TSV output files
        :param corpus_path: str or Path, path to the root directory or file(s) relevant to the corpus
        :param has_oracle: True if the corpus has gold standard annotations for lemmas and morphological analysis
        :param filters: list of str or None, used to select a subcorpus
        """
        self.language = language
        self.corpus_name = corpus_name
        self.corpus_path = Path(corpus_path)
        self.has_oracle = has_oracle
        self.filters=filters
        self.doc_count = 0
        self.token_count = 0

    @abstractmethod
    def document_generator(self):
        """A generator function to iterate over documents in the corpus.
        Yields document id and the entry point to access the
        document (e.g. XML root or file path)
        Should also increment doc_count
        """

    @abstractmethod
    def token_generator(self, document_root):
        """A generator function to iterate over all tokens in the given document.
        Yields the surface form, the oracle form and the morphological analysis.
        Should also increment token_count.
        """

    def get_average_doc_length(self):
        """Returns the average number of kept tokens per document.
        """
        return self.token_count/self.doc_count

    def print_corpus_statistics(self):
        """Prints the number of documents, number of token counts and average doc length for the corpus
        """
        print(self.corpus_name, "corpus statistics:")
        if self.filters is not None:
            print("\tSubcorpus filters applied:", self.filters)
        print("\tTotal documents:", self.doc_count)
        print("\tTotal tokens:", self.token_count)
        print("\tAverage doc length by tokens:", self.get_average_doc_length())


class RussianNationalCorpusParser(CorpusParser):
    """Iterate over document files in the Russian National Corpus.
    The corpus is structured as file tree:
    RNC_million (root dir):
    ├── TABLES: documentation and tag set lists
    └── TEXTS: subcorpora - each of these folders contains xml files, 1 per document
        ├── blogs_2013
        ├── fiction
        ├── public
        ├── science
        └── speech
    """
    def __init__(self, rnc_root, filters=None):
        """
        :param rnc_root: str or Path, The root path of the RNC, a folder containing subfolders 'TABLES' and 'TEXT'
        :param filters: list of str, The names of subcorpus folders you'd like to process. Use filters=None to keep everything.
        """
        super().__init__("ru", RNC, rnc_root, has_oracle=True, filters=filters)

    def document_generator(self):
        """Generator function to iterate over all the documents in the corpus
        that match the subcorpus filters.
        Returns the document id and path to a single XML file
        """
        print("Starting iteration of the Russian National Corpus")
        for subcorpus in Path(self.corpus_path / 'TEXTS').iterdir():
            if self.filters is None or (subcorpus.name in self.filters and subcorpus.is_dir()):
                print("Parsing subcorpus:", subcorpus.name)
                for xml_file in subcorpus.iterdir():
                    doc_id = xml_file.stem
                    self.doc_count+=1
                    yield doc_id, xml_file

            else:
                print("Skipping subcorpus:", subcorpus.name)

    def token_generator(self, xml_path):
        """Generator function to iterate over all tokens in the XML file.
        Yields the surface form, the oracle form and the morphological analysis.
        If there is no oracle form or morphological analysis, these will be None.
        :param xml_path: str, full path to the xml file for a single document in RNC
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
                self.token_count+=1
                yield surface_form, AMBIGUOUS_ANALYSIS_SEP.join(lemmas), AMBIGUOUS_ANALYSIS_SEP.join(grs)

            except AttributeError as e:
                print(f"ERROR: Problem at tag with annotation tail '{annotation_tail}', lex '{lemmas}', gr '{grs}'")
                raise e


class OpenCorporaParser(CorpusParser):
    """Parses OpenCorpora from its single xml document format.
    Filtering is annoying, since the text XML tag is used to describe documents and publication sources.
    The publication genres in OpenCorpora are the <tag>Тип:genre</tag> tags, where genre can be one of the following:
    'newspaper', 'encyclopedia', 'blog', 'literature', 'nonfiction', 'legal'.
    Note that these are mapped to the corresponding Russian translation for parsing the XML.
    """
    # TODO Refactor iteration pieces to be clearer
    # Constants for the XML namespace
    # Yes, OpenCorpora uses the same values for different things, but let's name them differently here for clarity
    DOC_TAG = "text"
    META_TAG = "tag"
    ANNOTATION_TAG = "v"
    LEXEME_TAG = "l"
    GRAMMAR_TAG = "g"
    SLOT_ATTRIB_KEY = "v"
    LEMMA_ATTRIB_KEY = "t"
    TOKEN_TAG = "token"
    SURFACE_ATTRIB_KEY = "text"
    ID_ATTRIB_KEY = "id"
    CORPUS_KEY = "annotation" # Root of entire xml

    # Translations for genres in OpenCorpora
    filter_translations = {
        'газета':'newspaper',
        'энциклопедия':'encyclopedia',
        'блог':'blog',
        'художественная литература':'literature',
        'нонфикшен':'nonfiction',
        'юридические тексты':'legal'
    }

    def __init__(self, root_xml, filters=None):
        """
        :param root_xml: str, path to xml file
        :param filters: list of str, the publication genre tags for the subcorpus you'd like to process. Use filters=None to keep everything.
        """
        # These get updated as you iterate through the corpus, map doc id to genre
        self.doc_genres = {}
        # Store ids of leaf nodes (actual documents containing text)
        self.doc_ids = set()
        super().__init__("ru", OPENCORPORA, root_xml, has_oracle=True, filters=self.get_filters(filters))

    def get_filters(self, filters):
        """Checks that filters are valid.
        Returns approriate filters for this corpus parser.

        :param filters: list of str, the publication genre tags for the subcorpus you'd like to process. Use filters=None to keep everything.
        """
        if filters is not None:
            valid_filters = []
            for f in filters:
                downcase_f = f.lower()
                if downcase_f in self.filter_translations.values():
                    valid_filters.append(downcase_f)
                else:
                    print(f"WARNING: Unknown filter value: {f}")
            return valid_filters
        return filters

    def _determine_genres(self):
        """Iterate through once to collect all genre tags for all documents
        This is required due to the unpredictable nested structure of the corpus
        and because not all metadata is populated down to children.
        """
        # Maps ids of docs where genre isn't known to their parent's id
        unknown_genre_docs = {}
        # Documents that are parents don't have text
        all_parent_ids = set(['0'])
        # Track valid ids
        all_valid_ids = set()
        invalid_ids = set()

        print("Identifying genres and leaf node documents.")
        for _, elem in ET.iterparse(self.corpus_path, events=(START, )):
            if elem.tag == self.DOC_TAG:
                parent_id = str(elem.get('parent'))
                elem_id = str(elem.get(self.ID_ATTRIB_KEY))
                all_parent_ids.add(parent_id)
                all_valid_ids.add(elem_id)
                # Subcorpus nodes have a parent="0" attribute
                # For subcorpus nodes, check the genre
                if parent_id=='0':
                    genre = self._find_genre(elem)
                    if not genre:
                        all_valid_ids.remove(elem_id)
                        invalid_ids.add(elem_id)

                # Parent's genre is inherited by child
                elif parent_id in self.doc_genres:
                    self.doc_genres[elem_id] = self.doc_genres[parent_id]

                # Invalid parents invalidate children
                elif parent_id in invalid_ids:
                    print(f"WARNING: Invalid parent id '{parent_id}' for doc id '{elem_id}' while checking genre. Parent doesn't have valid genre.")
                    all_valid_ids.remove(elem_id)
                    invalid_ids.add(elem_id)

                # We haven't encountered some parent in tree yet, save parent id and check later
                else:
                    unknown_genre_docs[elem_id] = parent_id
            # Reclaim memory as you iterate through the XML tree
            elem.clear()

        # TODO clean up
        # Assign all documents with a genre or mark them as unknown permanently
        while len(unknown_genre_docs) > 0:
            # Iterate in reverse, since issue arises when parent_id > doc_id
            unknown_docs_iter = list(unknown_genre_docs.items())
            unknown_docs_iter.reverse()
            print("Determining genre for", len(unknown_docs_iter), "remaining document(s).")

            for doc_id, parent_id in unknown_docs_iter:
                if parent_id in self.doc_genres:
                    self.doc_genres[doc_id] = self.doc_genres[parent_id]
                    del unknown_genre_docs[doc_id]
                elif parent_id not in all_valid_ids or parent_id in invalid_ids:
                    # This is something very strange, throw it away
                    print(f"WARNING: Invalid parent id '{parent_id}' for doc id '{doc_id}' while checking genre. Parent doesn't have valid genre.")
                    del unknown_genre_docs[doc_id]
                    all_valid_ids.remove(doc_id)
                    invalid_ids.add(doc_id)

        # Update doc_ids to store only ids of leaf nodes
        self.doc_ids = all_valid_ids - all_parent_ids
        print("Finished determining leaf node documents.")

    def _find_genre(self, subcorpus_element):
        """Returns the genre found for a subcorpus and stores result in self.doc_genres.
        If no valid genre can be found, returns None.
        :param subcorpus_element: xml.etree.ElementTree.Element with 'text' tag and 'tag' children
        """
        name = subcorpus_element.get('name')
        elem_id = subcorpus_element.get(self.ID_ATTRIB_KEY)
        for tag in subcorpus_element.iter(tag=self.META_TAG):
            tag_text = tag.text.lower()
            if tag_text.startswith('тип:'):
                russian_genre = tag_text.split(":")[1]
                genre = self.filter_translations[russian_genre]
                self.doc_genres[elem_id] = genre
                print(f"Found subcorpus '{name}' with id '{elem_id}' and genre '{genre}'.")
                return genre

        print(f"WARNING: No genre found for subcorpus with id '{elem_id}' and name '{name}'.")
        return None

    def document_generator(self):
        """Generator function to iterate over all the documents in the corpus
        that match the subcorpus filters.
        Returns the document id and the document iterator that emits 'start' and 'end' events positioned at the beginning of the next document.
        See https://docs.python.org/3/library/xml.etree.elementtree.html#xml.etree.ElementTree.iterparse
        for details about the document iterator.
        """
        print("Starting iteration of OpenCorpora.")
        # This is a little inefficient, as it looks at some nodes mutiple times, but hopefully won't be too slow
        self._determine_genres()

        print("Starting iteration through leaf node documents.")
        doc_iter = ET.iterparse(self.corpus_path, events=(END, START,))
        event, elem = next(doc_iter)
        while not (event==END and elem.tag==self.CORPUS_KEY):
            # Each text tag represents a document or subcorpus
            # We need to pull out different information depending on which
            if elem.tag==self.DOC_TAG and event==START and self.ID_ATTRIB_KEY in elem.attrib:
                elem_id = str(elem.get(self.ID_ATTRIB_KEY))
                # Check it's a leaf node which contains text
                if elem_id in self.doc_ids:
                    if self.filters is None or (self.doc_genres[elem_id] in self.filters):
                        self.doc_count+=1
                        yield elem_id, doc_iter
            # Tags can be cleared if they aren't valid documents
            elif event==END:
                elem.clear()
            event, elem = next(doc_iter)

    def token_generator(self, doc_iter):
        """Generator fucntion to iterate over all tokens in the document's XML node.
        Return the surface form, the oracle form and the morphological analysis.

        :param doc_iter: an iterator positioned at the beginning of the relevant document (last element read was the 'text' start tag for the document)
        """
        # If this method raises an error, then the XML is malformed, so let's risk the StopIteration exception
        event, elem = next(doc_iter) # Should be first token opening tag
        # Loop until you reach the end of the document
        while not (elem.tag==self.DOC_TAG and event==END):
            # Examine each token
            if elem.tag==self.TOKEN_TAG and event==START:
                # As with RNC, you can have multiple lemmas and morphology slots for ambiguous forms
                surface_form = elem.get(self.SURFACE_ATTRIB_KEY)
                # Collect ambiguous forms
                oracle_form, morphology = self._parse_token_children(doc_iter)
                self.token_count+=1
                yield surface_form, oracle_form, morphology
            event, elem = next(doc_iter)

        # After a document has been parsed, it can be discarded
        elem.clear()

    def _parse_token_children(self, doc_iter):
        """Returns the lemmas and morphological analysis information gathered from
        the a 'token' tag's children.
        :param doc_iter: A ET iterparse generator pointing in the token tag
        """
        # As with RNC, you can have multiple lemmas and morphology slots for ambiguous forms
        lemmas = []
        morph_analyses = []
        event, elem = next(doc_iter)
        while not (elem.tag==self.TOKEN_TAG and event==END):
            if elem.tag==self.ANNOTATION_TAG and event==START:
                morph_slots = []
            # Each annotation consists of a lexeme dictionary form...
            elif elem.tag==self.LEXEME_TAG and event==START:
                lemmas.append(elem.get(self.LEMMA_ATTRIB_KEY))
            # ... and a bunch of grammatical slot information, each in its own tag
            elif elem.tag==self.GRAMMAR_TAG and event==START:
                morph_slots.append(elem.get(self.SLOT_ATTRIB_KEY))
            # After each annotation, collect all the grammatical information for a particular lexeme to a list
            elif elem.tag==self.ANNOTATION_TAG and event==END:
                morph_analyses.append(SLOT_SEP.join(morph_slots))
            event, elem = next(doc_iter)

        # Collect ambiguous forms
        oracle_form = AMBIGUOUS_ANALYSIS_SEP.join(lemmas)
        morphology = AMBIGUOUS_ANALYSIS_SEP.join(morph_analyses)
        return oracle_form, morphology

class TIGERDocument:
    """
    Convenience container for determining sentences in each TIGER document
    Luckily sentence ids are integers and are consecutive
    """
    def __init__(self, doc_id, first_sentence_id, last_sentence_id, xml_iter):
        """
        :param doc_id: str, the document id
        :param first_sentence_id: int, the id of the first sentence in document
        :param last_sentence_id: int, the id of the last in document
        :param xml_iter: ET iterparse object pointing at the first sentence in the document
        """
        self.doc_id = doc_id
        self.first_sentence_id = first_sentence_id
        self.last_sentence_id = last_sentence_id
        self.xml_iter = xml_iter


class TIGERCorpusParser(CorpusParser):
    """Parses the TIGER v2.2 corpus from a single XML file and the 'documents.tsv' file from TIGER 2.2.doc for the sentence to document mapping.

    Note that TIGER is from a single news source and genre, so it has no subcorpus filters.
    """
    # Constants for XML parsing
    SENTENCE_TAG = "s"
    TOKEN_TAG = "t"
    WORD_ATTRIB = "word"
    LEMMA_ATTRIB = "lemma"
    POS_ATTRIB = "pos"
    MORPH_ATTRIB = "morph"
    ID_ATTRIB = "id"
    CORPUS_TAG = "corpus"

    def __init__(self, xml_root, sentence_map_tsv):
        """
        :param xml_root: str, path to the xml file containing the entire TIGER corpus tokens
        :param sentence_map_tsv: str, path to the TSV file that has doc_id,sentence_id rows for each sentence in the corpus
        """
        self.sentence_map_tsv = sentence_map_tsv
        # Tracks the document you're currently reading
        super().__init__("de", TIGER, xml_root, has_oracle=True)

    def document_generator(self):
        """Generator for the documents in the corpus.
        """
        # Iterate until you get to the first sentence tag
        doc_iter = ET.iterparse(self.corpus_path, events=(START, END, ))
        advance_xml_iterator(doc_iter, self.SENTENCE_TAG, START, self.CORPUS_TAG)

        with open(self.sentence_map_tsv, newline='', encoding="utf-8") as f:
            prev_doc_id = None
            for doc_id, sentence_id in csv.reader(f, dialect='unix', delimiter="\t"):
                # If doc_id changes, you've found the last sentence and should yield the document
                if doc_id != prev_doc_id:
                    if prev_doc_id is not None:
                        self.doc_count+=1
                        yield prev_doc_id, TIGERDocument(prev_doc_id, first_sentence_id, prev_sentence_id, doc_iter)
                    prev_doc_id = doc_id
                    first_sentence_id = int(sentence_id)
                    # Advance the document iterator to the next start sentence tag for the next document
                    advance_xml_iterator(doc_iter, self.SENTENCE_TAG, START, self.CORPUS_TAG)
                prev_sentence_id = int(sentence_id)

            # Don't forget the last document
            if prev_doc_id is not None:
                self.doc_count+=1
                yield prev_doc_id, TIGERDocument(prev_doc_id, first_sentence_id, prev_sentence_id, doc_iter)

    def token_generator(self, document_root):
        """Yields the surface form, the oracle form and the morphological analysis.

        :param document_root: TIGERDocument object
        """
        current_sentence = document_root.first_sentence_id
        is_parsed_final_sentence = False
        while not is_parsed_final_sentence:
            event, elem = next(document_root.xml_iter)
            if event==START and elem.tag==self.TOKEN_TAG:
                self.token_count+=1
                yield self._parse_token_attributes(elem)
            if event==START and elem.tag==self.SENTENCE_TAG:
                current_sentence = self.get_sentence_id_from_attrib(elem.get(self.ID_ATTRIB))
            elif event==END and elem.tag==self.SENTENCE_TAG and current_sentence==document_root.last_sentence_id:
                # The document is finished
                is_parsed_final_sentence = True
            if event==END:
                elem.clear()

    def _parse_token_attributes(self, token_element):
        """Returns the surface form, lemma and morphological analysis for a
        token entry from TIGER XML.
        :param token_element: A token tag xml entry with attributes
        """
        # Morphological analyses aren't ambiguous in TIGER, so we just have to check one per word
        surface_form = token_element.get(self.WORD_ATTRIB)
        pos = token_element.get(self.POS_ATTRIB)
        # USE comma separation of slots, to match RNC
        morph = token_element.get(self.MORPH_ATTRIB).replace(".", SLOT_SEP)
        lemma = token_element.get(self.LEMMA_ATTRIB)
        full_morph_analysis = SLOT_SEP.join([pos, morph])
        return surface_form, lemma, full_morph_analysis

    def get_sentence_id_from_attrib(self, attribute_value):
        """Returns the sentence id as an integer from the XML attribute value
        :param attribute_value: str, the value of the sentence's 'id' attribute from xml
        """
        return int(attribute_value.strip('s'))


class CorpusPreprocessor:
    # TODO Add min_doc_tokens to set lower bound on document length by number of tokens
    # TODO Add corpus stats computed by what's kept for Mallet, rather than full original corpus
    def __init__(self, corpus_parser, output_dir, use_oracle=True, use_truncation_stemmers=True, truncation_settings=None):
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

        print("Using processing methods:", list(self.output_tsvs.keys()))

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
        print(f"{self.output_tsvs[RAW].num_docs} documents processed.")
        print(f"Finished processing corpus {self.corpus_parser.corpus_name}.")
        self.corpus_parser.print_corpus_statistics()

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

    :param corpus_path: list of str or Path, paths to the root directory or file(s) relevant to the corpus
    :param output_dir: The output directory to write all TSV files to
    """
    if corpus_name==RNC:
        # Just keep the news section of RNC for now
        return RussianNationalCorpusParser(corpus_path[0], filters=['public'])
    elif corpus_name==OPENCORPORA:
        return OpenCorporaParser(corpus_path[0], filters=['newspaper'])
    elif corpus_name==TIGER:
        return TIGERCorpusParser(corpus_path[0], corpus_path[1])
    else:
        raise ValueError(f"Invalid corpus choice: {corpus_name}")


def main(corpus_name, corpus_in, output_dir):
    corpus_parser = get_corpus_parser(corpus_name, corpus_in)
    preprocessor = CorpusPreprocessor(corpus_parser, output_dir)
    preprocessor.parse_corpus()


parser = argparse.ArgumentParser(
description = """Preprocessing for specific corpora formats, including
stopword removal, language and appropriate stemming.
Writes out preprocessed corpora to Mallet's TSV
format and also writes out a TSV with oracular morphological information
for terms when this is available. SpaCy's stop word list is used.
Currently supported corpora:
 - Russian National Corpus (RNC)
 - OpenCorpora (single XML file version)
 - TIGER v2.2""",
   formatter_class= argparse.RawTextHelpFormatter)
parser.add_argument("--corpus_in", nargs='+', required=True,
    help="""Path to corpus input file or files.
Could be a single file or single directory depending on the corpus.
These are expected input formats for each corpus (pass to --corpus_in argument):
- RNC: Root directory to the corpus (folder containing 'TABLES' and 'TEXTS' directories)
- OpenCorpora: Path to the single XML file version of the corpus
- TIGER v2.2: 2 arguments required 1) XML file for the full corpus 2) TSV file containing document and sentence id mapping""")
parser.add_argument("--corpus_name", type=str, required=True,
    choices=[RNC, OPENCORPORA, TIGER],
    help="Identifier for the corpus." )
parser.add_argument("--output_dir", required=True,
    help="Path to output directory for all Mallet TSV files.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.corpus_name, args.corpus_in, args.output_dir)