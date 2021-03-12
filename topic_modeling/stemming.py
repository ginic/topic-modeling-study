# coding=utf-8
'''Shared functionalities for stemming and lemmatization

# TODO Pymorphy2
'''
import nltk.stem.snowball as nltkstem
import pymorphy2
import pymystem3
import stanza

import topic_modeling.tokenization as tokenization

# Desired stanza Russian modeling settings
STANZA_SETTINGS = 'tokenize,lemma'
STANZA_PACKAGE = 'syntagrus'

PYMYSTEM_ANALYSIS = 'analysis'
PYMYSTEM_LEX = 'lex'
PYMYSTEM_TEXT = 'text'

class StemmingError(Exception):
    '''Raised when underlying stemmers do not behave as expected'''
    pass

class StanzaLemmatizer:
    '''Wrapper around the Stanza/Stanford CoreNLP lemmatizer for Russian
    '''
    def __init__(self):
        stanza.download('ru', processors=STANZA_SETTINGS, package=STANZA_PACKAGE)
        self.pipeline = stanza.Pipeline('ru',processors=STANZA_SETTINGS,
                                        package=STANZA_PACKAGE)

    def lemmatize(self, text, keep_punct=False):
        '''Returns list of (word, lemma) pairs for each word in the given text.
        Stanza's sentence breaking and tokenization is used.

        :param text: str, Russian text to process
        :param keep_punct: True to keep tokens/lemmas that are just punctuation
        '''
        result = list()
        doc = self.pipeline(text)
        for sent in doc.sentences:
            for word in sent.words:
                # We don't want any tokens that are only puctuation
                clean_word = tokenization.clean_punctuation(word.text)
                if clean_word != '' or keep_punct:
                    result.append((word.text, word.lemma))
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
                result.append((t,s))
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
    def __init__(self):
        '''Instantiate Mystem
        '''
        self.mystem = pymystem3.Mystem()

    def lemmatize(self, text, keep_unanalyzed_tokens=False):
        '''Returns a list (token, lemma) pairs determined for the text
        by Mystem.

        :param text: str, text to tokenize and lemmatize.
        :param keep_unanalyzed_tokens: True to also return non-whitespace
            tokens that have no morphological analysis, such as numbers or
            punctuation
        '''
        result = list()
        analysis = self.mystem.analyze(text)
        for a in analysis:
            token = a[PYMYSTEM_TEXT]
            if PYMYSTEM_ANALYSIS in a:
                lexes = a[PYMYSTEM_ANALYSIS]
                if len(lexes) > 1:
                    raise StemmingError(f"Mystem returned multiple analyses, only 1 expected: {a}")
                result.append((token, lexes[0][PYMYSTEM_LEX]))
            elif keep_unanalyzed_tokens and not t.isspace() and t!='':
                result.append(token)
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
            result.append((t, lemma))

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
        return [(t, t[:self.num_chars]) for t in tokens]

