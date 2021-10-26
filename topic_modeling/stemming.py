# coding=utf-8
"""Shared functionalities for stemming and lemmatization.
The main/script will lemmatize or stem a given input TSV in Mallet format
and produced another TSV in Mallet format with appropriately normalized text
and a TSV with counts of each (token,lemma) pair by author.
"""
# TODO language options added beyond russian for script behaviour
# TODO Add morphological analyzer class as mixin for stanza, spacy and pymystem
from abc import ABC, abstractmethod
import argparse
import collections
import time
import traceback

import nltk.stem.snowball as nltkstem
import pandas as pd
import pymorphy2
import pymystem3
import spacy
import stanza

import topic_modeling.tokenization as tokenization

# choices for stemmers
PYMORPHY = 'pymorphy2'
PYMYSTEM = 'pymystem3'
SNOWBALL = 'snowball'
STANZA = 'stanza'
TRUNCATE = 'truncate'
SPACY = "spacy"
STEMMER_CHOICES = [PYMORPHY, PYMYSTEM, SNOWBALL, STANZA, TRUNCATE, SPACY]

# Desired stanza modeling settings
STANZA_SETTINGS = 'tokenize,pos,lemma'
# Note regarding German stanza models: The 'gsd' model is the default model, but it expands contractions to multiword tokens
# (e.g. 'zum' to 'zu dem' rather than 'zu'), which could cause headaches.
STANZA_PACKAGE = {'ru':'syntagrus', 'de':'hdt'}

# Pymystem keys
PYMYSTEM_ANALYSIS = 'analysis'
PYMYSTEM_LEX = 'lex'
PYMYSTEM_TEXT = 'text'

# Container for tokens and their corresponding stem or lemma
NormalizedToken = collections.namedtuple('NormalizedToken', 'token normalized')


class ByAuthorStemCounts:
    """Maps author to term frequencies for easily generating by-author statistics.
    All tokens and lemmas are downcased, since that's what Mallet does
    """

    def __init__(self):
        self.author_map = collections.defaultdict(collections.Counter)

    def update(self, author, token_stem_pairs):
        """Updates frequency counts of this author for the given
        list of NormalizedToken pairs

        :param author: str, name of author to add stats for
        :param token_stem_pairs: list of NormalizedToken tuples
        """
        self.author_map[author].update([(t[0].lower(), t[1].lower()) for t in token_stem_pairs])

    def to_dataframe(self):
        """Returns the author_map as a pandas dataframe
        with columns for author, original token, normalized token
        (stem or lemma) and counts
        """
        records = list()
        for author in self.author_map:
            for token_pair, count in self.author_map[author].items():
                records.append((author, token_pair[0], token_pair[1], count))

        return pd.DataFrame.from_records(records,
                    columns=['author', 'token','normalized','count'])


class StemmingError(Exception):
    """Raised when underlying stemmers do not behave as expected"""
    pass


class AbstractLemmatizer(ABC):
    """Abstract class for defining behaviour of Stemmers and Lemmatizers
    """
    @abstractmethod
    def lemmatize(self, text):
        """Tokenizes text, then returns a list of (token, lemma) pairs for tokens in text.

        :param text: str, text to process
        """

    @abstractmethod
    def single_term_lemma(self, word):
        """Returns the lemma of a single word as a string. Beware this can return empty strings in some cases.

        :param word: str, single word to get lemma for
        """


class StanzaLemmatizer(AbstractLemmatizer):
    """Wrapper around the Stanza/Stanford CoreNLP lemmatizer.
    """
    def __init__(self, keep_punct=False, language='ru'):
        """Instantiates Stanza lemmatizer and ensures 'ru' models are downloaded

        :param keep_punct: True to keep tokens/lemmas that are just punctuation
        :param language: str, two letter language code
        """
        stanza_package = STANZA_PACKAGE[language]
        stanza.download(language, processors=STANZA_SETTINGS, package=stanza_package)
        self.pipeline = stanza.Pipeline(language,processors=STANZA_SETTINGS,
                                        package=stanza_package)
        self.keep_punct = keep_punct

    def lemmatize(self, text):
        """Returns list of (word, lemma) pairs for each word in the given text.
        Stanza's sentence breaking and tokenization is used.

        :param text: str, text to process
        """
        result = list()
        doc = self.pipeline(text)
        for sent in doc.sentences:
            for word in sent.words:
                # We don't want any tokens that are only puctuation
                clean_word = tokenization.clean_punctuation(word.text)
                if clean_word != '' or self.keep_punct:
                    result.append(NormalizedToken(word.text, word.lemma))
        return result

    def single_term_lemma(self, word):
        """Returns the lemma of a single word as a string. Beware this can return empty strings in some cases.

        :param word: str, single word to get lemma for
        """
        doc = self.pipeline(word)
        for sent in doc.sentences:
            for word in sent.words:
                return word.lemma



class SnowballStemmer(AbstractLemmatizer):
    """Wrapper around NLTK's implementation of the Snowball Stemmer,
    which uses an improved Porter stemming algorithm.
    http://snowball.tartarus.org/algorithms/russian/stemmer.html
    """
    def __init__(self, tokenizer=None, language='russian'):
        """Instantiate NLTK Snowball stemmer. Default tokenizer
        is RegexTokenizer with WORD_TYPE_NO_DIGITS_TOKENIZATION

        :param tokenizer: object with a tokenize(str) method
        """
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = tokenization.RegexTokenizer()

        self.stemmer = nltkstem.SnowballStemmer(language)

    def lemmatize(self, text):
        """Tokenizes and stems each word in the given text.
        Returns list of (word, lemma) pairs.
        :param text: str, Russian text to process
        """
        result = list()
        tokens = self.tokenizer.tokenize(text)
        for t in tokens:
            s = self.stemmer.stem(t)
            if not s.isspace() and s!='':
                result.append(NormalizedToken(t,s))
        return result

    def single_term_lemma(self, word):
        """Returns the lemma of a single word as a string. Beware this can return empty strings in some cases.

        :param word: str, single word to get lemma for
        """
        return self.stemmer.stem(word)


class Pymystem3Lemmatizer(AbstractLemmatizer):
    """Wrapper around Pymystem3 implementation. It supports Russian, Polish
    and English lemmatization.
    Note that Mystem does its own tokenization.
    The analyze function returns one best lemma prediction. Example:
    >>>self.mystem.analyze("это предложение")
    >>>[{'analysis': [{'lex': 'этот', 'wt': 0.05565618415, 'gr': 'APRO=(вин,ед,сред|им,ед,сред)'}], 'text': 'это'},
       {'text': ' '}, {'analysis': [{'lex': 'предложение', 'wt': 1, 'gr': 'S,сред,неод=(вин,ед|им,ед)'}], 'text': 'предложение'}]
    """
    def __init__(self, keep_unanalyzed=False):
        """Instantiate Mystem wrapper

        :param keep_unanalyzed: True to also return non-whitespace and
            non-punctuation tokens that have no morphological analysis, like numbers
        """
        self.mystem = pymystem3.Mystem()
        self.keep_unanalyzed = keep_unanalyzed

    def lemmatize(self, text):
        """Returns a list (token, lemma) pairs determined for the text
        by Mystem.

        :param text: str, text to tokenize and lemmatize.

        """
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
                clean_token = tokenization.clean_punctuation(token)
                if not clean_token.isspace() and clean_token!='':
                    result.append(NormalizedToken(token, token))
        return result

    def single_term_lemma(self, word):
        """Returns the lemma of a single word as a string. Beware this can return empty strings in some cases. The `keep_unanalyzed' flag is ignored for this method.

        :param word: str, single word to get lemma for
        """
        analysis = self.mystem.analyze(word)
        result = ''
        # Pymystem always ends its analysis with a new line character,
        # so result will be at least 2 elements. Some words, particularly
        # those with hyphens might be split up into multiple lemmas,
        # but we want to join them and return a single string
        for a in analysis[:-1]:
            if PYMYSTEM_ANALYSIS in a and len(a[PYMYSTEM_ANALYSIS]) > 0:
                # Keep the highest scoring result
                lexes = a[PYMYSTEM_ANALYSIS]
                result += lexes[0][PYMYSTEM_LEX].strip()
            else:
                result += a[PYMYSTEM_TEXT].strip()

        # Catch things that pymystem doesn't analyze well
        if result == '':
            return word

        return result


class Pymorphy2Lemmatizer(AbstractLemmatizer):
    """Wrapper around Pymorphy2Lemmatizer. Default tokenizer
    is RegexTokenizer with WORD_TYPE_NO_DIGITS_TOKENIZATION

    :param tokenizer: object with a tokenize(str) method
    """
    def __init__(self, tokenizer=None):
        """Instantiates pymorphy2 and specified tokenization.
        """
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = tokenization.RegexTokenizer()

        self.analyzer = pymorphy2.MorphAnalyzer()

    def lemmatize(self, text):
        """Return list of (token, lemma) in lemmatized with pymorphy2

        :param text: str, text to tokenize and lemmatize
        """
        result = list()
        tokens = self.tokenizer.tokenize(text)
        for t in tokens:
            parses = self.analyzer.parse(t)
            # pymorphy isn't context aware, just take most likely form
            lemma = parses[0].normal_form
            result.append(NormalizedToken(t, lemma))

        return result

    def single_term_lemma(self, word):
        """Returns the lemma of a single word as a string. Beware this can return empty strings in some cases.

        :param word: str, single word to get lemma for
        """
        return self.analyzer.parse(word)[0].normal_form


class TruncationStemmer(AbstractLemmatizer):
    """A naive strategy to stem by keeping the first num_chars number of
    characters in a word
    """
    def __init__(self, tokenizer=None, num_chars=5):
        """Instantiate TrucationStemmer. Default tokenizer
        is RegexTokenizer with WORD_TYPE_NO_DIGITS_TOKENIZATION

        :param tokenizer: an object with a tokenize(str) method
        :param num_chars: int, word initial characters to keep, defaults to 5
        """
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = tokenization.RegexTokenizer()
        self.num_chars = num_chars

    def lemmatize(self, text):
        """Returns list of (token, stems) pairs after tokenizing text
        and trucating each token.

        :param text: str, text to tokenize and stem
        """
        tokens = self.tokenizer.tokenize(text)
        return [NormalizedToken(t, t[:self.num_chars]) for t in tokens]

    def single_term_lemma(self, word):
        """Returns the lemma of a single word as a string. Beware this can return empty strings in some cases.

        :param word: str, single word to get lemma for
        """
        return word[:self.num_chars]


class SpaCyLemmatizer(AbstractLemmatizer):
    """Load a spaCy model for lemmatization
    """
    def __init__(self, keep_punct=False, language_model='de_core_news_lg'):
        """Instantiates spaCy model.
        This will throw an error if you don't have the language model downloaded.
        """
        self.nlp = spacy.load(language_model)
        self.keep_punct = keep_punct

    def lemmatize(self, text):
        """Returns a list of (word, lemma) pairs for each word in the given text. Spacy's sentence breaking and tokenization is used.

        :param text: str, text to process
        """
        results = []
        spacy_doc = self.nlp(text)
        for token in spacy_doc:
            if not token.is_punct or self.keep_punct:
                results.append(NormalizedToken(token.text, token.lemma_))

        return results

    def single_term_lemma(self, word):
        """Returns the lemma of a single word as a string. Beware this can return empty strings in some cases.

        :param word: str, single word to get lemma for
        """
        return self.nlp(word)[0].lemma_


def pick_lemmatizer(choice):
    """Returns a lemmatizer object with default settings corresponding to
    user input choice
    .. TODO: currently only returns Russian stemmers
    :param choice: str, defined choice of lemmatizer to use
    """
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

def get_language_specific_stemmers(language):
    """Returns a dictionary of {"stemmer_name": instance of AbstractLemmatizer} containing
    lemmatizers and stemmers appropriate for the given language.

    :param language: str, language name or code
    """
    lang = language.lower()
    if lang in ["russian", "ru"]:
        stemmers = { PYMYSTEM: Pymystem3Lemmatizer(),
                     STANZA: StanzaLemmatizer(language='ru'),
                     SNOWBALL: SnowballStemmer(language='russian')
                    }
    elif lang in ["german", "de"]:
        stemmers = { SNOWBALL :SnowballStemmer(language='german'),
                     STANZA: StanzaLemmatizer(language='de'),
                     SPACY: SpaCyLemmatizer()
                    }
    else:
        raise ValueError(f"Not a valid language choice: {language}")

    return stemmers


def main(tsv_in, text_col, tsv_out, count_tsv, lemmatizer, author_col):
    """

    :param tsv_in: str, path to input tsv file
    :param text_col: int, index of column to stem/lemmatize
    :param tsv_out: str, path to output tsv file for normalized text
    :param count_tsv: str, path to output tsv file for counts of token, lemma pairs by author
    :param lemmatizer: object with lemmatize(text) method
    :param author_col: int, index of column with author label
    """
    print("Lemmatizing", tsv_in)
    start = time.perf_counter()
    errors = 0
    stem_counter = ByAuthorStemCounts()
    docs_in = open(tsv_in, 'r', encoding='utf-8')
    docs_out = open(tsv_out, 'w', encoding='utf-8')
    line_count = 1
    for row in docs_in:
        if line_count % 10 == 0:
            print("Reading line", line_count)
        try:
            split_row = row.strip().split('\t')
            # Find lemmas and write to output
            text = split_row[text_col]
            token_lemma_pairs = lemmatizer.lemmatize(text)
            lemmatized_text = " ".join([p.normalized for p in token_lemma_pairs])
            output_row = split_row[0:text_col] + [lemmatized_text] + split_row[text_col+1:]
            docs_out.write("\t".join(output_row) + "\n")
            # Update counts
            author = split_row[author_col]
            stem_counter.update(author, token_lemma_pairs)
            line_count += 1
        except Exception as e:
            errors +=1
            print("Falure at line", line_count)
            print("Row text:")
            print(row)
            traceback.print_exc()
            print("Document will be skipped.")
    docs_in.close()
    docs_out.close()
    end = time.perf_counter()
    print(f"Lemmatization took {end-start:0.2f} seconds")
    print(f"Failure to process", errors, "document(s)")
    print(f"Writing out lemma counts by author to", count_tsv)
    stem_counter.to_dataframe().to_csv(count_tsv, sep="\t", index=False)


parser = argparse.ArgumentParser(
    description="Lemmatizes or stems a given TSV. Normalizes text in the specified column, copies over other columns.")
parser.add_argument('tsv_in',
    help='Path to input TSV. Reading expects no escaping, just splits on every tab character')
parser.add_argument('tsv_out', help="Path to desired TSV output file for stemmed/lemmatized text in Mallet format.")
parser.add_argument('count_tsv', help="Path to tsv for storing counts of token, lemma pairs by author")
parser.add_argument('--col', '-c',
    help='Index of column to tokenize, default is 2',
    type=int,
    default=2)
parser.add_argument('--author-col', '-a',
    help='Index of column with author metadata, default is 1',
    type=int,
    default=1
)
parser.add_argument('--lemmatizer', '-l',
    help='Choice of stemmer/lemmatizer',
    choices=STEMMER_CHOICES)


if __name__ == "__main__":
    args = parser.parse_args()
    print("Lemmatization method:", args.lemmatizer)
    print(args)
    main(args.tsv_in, args.col, args.tsv_out, args.count_tsv,
        pick_lemmatizer(args.lemmatizer), args.author_col)