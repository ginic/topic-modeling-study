# coding=utf-8
'''Shared functionalities for stemming and lemmatization.
The main/script will lemmatize or stem a given input TSV in Mallet format.

# TODO output token,lemma,count to tsv/pickle to track lemmas/word types
'''
import argparse
import csv
from collections import namedtuple
import time

import nltk.stem.snowball as nltkstem
import pymorphy2
import pymystem3
import stanza

import topic_modeling.tokenization as tokenization

# choices for stemmers
PYMORPHY = 'pymorphy2'
PYMYSTEM = 'pymystem3'
SNOWBALL = 'snowball'
STANZA = 'stanza'
TRUNCATE = 'truncate'
STEMMER_CHOICES = [PYMORPHY, PYMYSTEM, SNOWBALL, STANZA, TRUNCATE]

# Desired stanza Russian modeling settings
STANZA_SETTINGS = 'tokenize,lemma'
STANZA_PACKAGE = 'syntagrus'

PYMYSTEM_ANALYSIS = 'analysis'
PYMYSTEM_LEX = 'lex'
PYMYSTEM_TEXT = 'text'

# Container for tokens and their corresponding stem or lemma
NormalizedToken = namedtuple('NormalizedToken', 'token normalized')

class StemmingError(Exception):
    '''Raised when underlying stemmers do not behave as expected'''
    pass


class StanzaLemmatizer:
    '''Wrapper around the Stanza/Stanford CoreNLP lemmatizer for Russian
    '''
    def __init__(self, keep_punct=False):
        '''Instantiates Stanza lemmatizer and ensures 'ru' models are downloaded

        :param keep_punct: True to keep tokens/lemmas that are just punctuation
        '''
        stanza.download('ru', processors=STANZA_SETTINGS, package=STANZA_PACKAGE)
        self.pipeline = stanza.Pipeline('ru',processors=STANZA_SETTINGS,
                                        package=STANZA_PACKAGE)
        self.keep_punct = keep_punct

    def lemmatize(self, text, keep_punct=False):
        '''Returns list of (word, lemma) pairs for each word in the given text.
        Stanza's sentence breaking and tokenization is used.

        :param text: str, Russian text to process
        '''
        result = list()
        doc = self.pipeline(text)
        for sent in doc.sentences:
            for word in sent.words:
                # We don't want any tokens that are only puctuation
                clean_word = tokenization.clean_punctuation(word.text)
                if clean_word != '' or self.keep_punct:
                    result.append(NormalizedToken(word.text, word.lemma))
        return result


class SnowballStemmer:
    '''Wrapper around NLTK's implementation of the Snowball Stemmer,
    which uses an improved Porter stemming algorithm.
    http://snowball.tartarus.org/algorithms/russian/stemmer.html
    '''
    def __init__(self, tokenizer=None):
        '''Instantiate NLTK Snowball stemmer. Default tokenizer
        is RegexTokenizer with WORD_TYPE_TOKENIZATION

        :param tokenizer: object with a tokenize(str) method
        '''
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = tokenization.RegexTokenizer()

        self.stemmer = nltkstem.SnowballStemmer('russian')

    def lemmatize(self, text):
        '''Tokenizes and stems each word in the given text.
        Returns list of (word, lemma) pairs.

        :param text: str, Russian text to process
        '''
        result = list()
        tokens = self.tokenizer.tokenize(text)
        for t in tokens:
            s = self.stemmer.stem(t)
            if not s.isspace() and s!='':
                result.append(NormalizedToken(t,s))
        return result


class Pymystem3Lemmatizer:
    '''Wrapper around Pymystem3 implementation. It supports Russian, Polish
    and English lemmatization.
    Note that Mystem does its own tokenization.
    The analyze function returns one best lemma preduction. Example:
    >>>self.mystem.analyze("это предложение")
    >>>[{'analysis': [{'lex': 'этот', 'wt': 0.05565618415, 'gr': 'APRO=(вин,ед,сред|им,ед,сред)'}], 'text': 'это'},
       {'text': ' '}, {'analysis': [{'lex': 'предложение', 'wt': 1, 'gr': 'S,сред,неод=(вин,ед|им,ед)'}], 'text': 'предложение'}]
    '''
    def __init__(self, keep_unanalyzed=True):
        '''Instantiate Mystem wrapper

        :param keep_unanalyzed: True to also return non-whitespace and
            non-punctuation tokens that have no morphological analysis, like numbers
        '''
        self.mystem = pymystem3.Mystem()
        self.keep_unanalyzed = keep_unanalyzed

    def lemmatize(self, text):
        '''Returns a list (token, lemma) pairs determined for the text
        by Mystem.

        :param text: str, text to tokenize and lemmatize.

        '''
        result = list()
        analysis = self.mystem.analyze(text)
        for a in analysis:
            token = a[PYMYSTEM_TEXT]
            if PYMYSTEM_ANALYSIS in a and len(a[PYMYSTEM_ANALYSIS]) > 0:
                # Keep the highest scoring result
                lexes = a[PYMYSTEM_ANALYSIS]
                result.append(NormalizedToken(token, lexes[0][PYMYSTEM_LEX]))
            elif self.keep_unanalyzed:
                # Don't keep tokens that are only spaces or only punctuation
                clean_token = tokenization.clean_punctuation(t)
                if not clean_token.isspace() and clean_token!='':
                    result.append(NormalizedToken(token, token))
        return result


class Pymorphy2Lemmatizer:
    '''Wrapper around Pymorphy2Lemmatizer. Default tokenizer
    is RegexTokenizer with WORD_TYPE_TOKENIZATION

    :param tokenizer: object with a tokenize(str) method
    '''
    def __init__(self, tokenizer=None):
        '''Instantiates pymorphy2 and specified tokenization.
        '''
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = tokenization.RegexTokenizer()

        self.analyzer = pymorphy2.MorphAnalyzer()

    def lemmatize(self, text):
        '''Return list of (token, lemma) in lemmatized with pymorphy2

        :param text: str, text to tokenize and lemmatize
        '''
        result = list()
        tokens = self.tokenizer.tokenize(text)
        for t in tokens:
            parses = self.analyzer.parse(t)
            # pymorphy isn't context aware, just take most likely form
            lemma = parses[0].normal_form
            result.append(NormalizedToken(t, lemma))

        return result

class TruncationStemmer:
    '''A naive strategy to stem by keeping the first num_chars number of
    characters in a word
    '''
    def __init__(self, tokenizer=None, num_chars=5):
        '''Instantiate TrucationStemmer. Default tokenizer
        is RegexTokenizer with WORD_TYPE_TOKENIZATION

        :param tokenizer: an object with a tokenize(str) method
        :param num_chars: int, word initial characters to keep, defaults to 5
        '''
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = tokenization.RegexTokenizer()
        self.num_chars = num_chars

    def lemmatize(self, text):
        '''Returns list of (token, stems) pairs after tokenizing text
        and trucating each token.

        :param text: str, text to tokenize and stem
        '''
        tokens = self.tokenizer.tokenize(text)
        return [NormalizedToken(t, t[:self.num_chars]) for t in tokens]


def pick_lemmatizer(choice):
    '''Returns a lemmatizer object with default settings corresponding to
    user input choice

    :param choice: str, defined choice of lemmatizer to use
    '''
    if choice==PYMORPHY:
        return Pymorphy2Lemmatizer()
    elif choice==PYMYSTEM:
        return Pymystem3Lemmatizer()
    elif choice==SNOWBALL:
        return SnowballStemmer()
    elif choice==STANZA:
        return StanzaLemmatizer()
    elif choice==TRUNCATE:
        return TruncationStemmer()
    else:
        raise ValueError(f"Stemmer choice '{choice}' is undefined")


def main(tsv_in, text_col, tsv_out, lemmatizer):
    '''

    :param tsv_in: str, path to input tsv file
    :param text_col: int, index of column to stem/lemmatize
    :param tsv_out: str, path to output tsv file
    :param lemmatizer: object with lemmatize(text) method
    '''
    print("Lemmatizing", tsv_in)
    start = time.perf_counter()
    docs_in = open(tsv_in, 'r', encoding='utf-8')
    docs_out = open(tsv_out, 'w', encoding='utf-8')
    tsv_reader = csv.reader(docs_in, delimiter='\t')
    tsv_writer = csv.writer(docs_out, delimiter='\t')
    for row in tsv_reader:
        if tsv_reader.line_num % 10 == 0:
            print("Reading line", tsv_reader.line_num)
        text = row[text_col]
        token_lemma_pairs = lemmatizer.lemmatize(text)
        lemmatized_text = " ".join([p.normalized for p in token_lemma_pairs])
        output_row = row[0:text_col] + [lemmatized_text] + row[text_col+1:]
        tsv_writer.writerow(output_row)

    docs_in.close()
    docs_out.close()
    end = time.perf_counter()
    print(f"Lemmatization took {end-start:0.2f} seconds")


parser = argparse.ArgumentParser(
    description="Lemmatizes or stems a given TSV. Normalizes text in the specified column, copies over other columns")
parser.add_argument('tsv_in',
    help='Path to input TSV format.')
parser.add_argument('tsv_out', help="Path to desired TSV output file.")
parser.add_argument('--col', '-c',
    help='Index of column to tokenize',
    type=int,
    default=2)
parser.add_argument('--lemmatizer', '-l',
    help='Choice of stemmer/lemmatizer',
    choices=STEMMER_CHOICES)


if __name__ == "__main__":
    args = parser.parse_args()
    print("Lemmatization method:", args.lemmatizer)
    print(args)
    main(args.tsv_in, args.col, args.tsv_out,
        pick_lemmatizer(args.lemmatizer))