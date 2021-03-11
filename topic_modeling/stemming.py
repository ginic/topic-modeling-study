# coding=utf-8
'''Shared functionalities for stemming and lemmatization

# TODO truncation 'stemmer' with specified max character length
# TODO NLTK Russian Porter (snowball) stemmer
# TODO Pymystem (POS tag sensitive)
# TODO Pymorphy2
'''
import nltk.stem.snowball as nltkstem
import stanza

import topic_modeling.tokenization as tokenization

# Desired stanza Russian modeling settings
STANZA_SETTINGS='tokenize,lemma'
STANZA_PACKAGE='syntagrus'

class StanzaLemmatizer:
    '''Wrapper around the Stanza/Stanford CoreNLP lemmatizer for Russian
    '''
    def __init__(self):
        stanza.download('ru', processors=STANZA_SETTINGS, package=STANZA_PACKAGE)
        self.pipeline = stanza.Pipeline('ru',processors=STANZA_SETTINGS,
                                        package=STANZA_PACKAGE)

    def lemmatize(self, text):
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
                if clean_word != '':
                    result.append((word.text, word.lemma))
        return result


class SnowballStemmer:
    '''Wrapper around NLTK's implementation of the Snowball Stemmer,
    which uses an improved Porter stemming algorithm.
    http://snowball.tartarus.org/algorithms/russian/stemmer.html
    '''
    def __init__(self, tokenizer):
        '''Instantiate NLTK Snowball stemmer

        :param tokenizer: object with a tokenize(str)
        '''
        self.tokenizer = tokenizer
        self.stemmer = nltkstem.SnowballStemmer('russian')

    def lemmatize(self, text):
        '''Tokenizes and stems each word in the given text.
        Returns list of (word, lemma) pairs.

        :param text: str, Russian text to process
        '''
        result = list()
        # TODO
        return result


class PymystemLemmatizer:
    '''Wrapper around Pymystem implementation.
    '''
    def __init__(self):
        '''TODO
        '''
        pass


class Pymorphy2Lemmatizer:
    '''Wrapper around Pymorphy2Lemmatizer
    '''
    def __init__(self):
        '''TODO
        '''
        pass


class TruncationStemmer:
    '''A naive strategy to stem by keeping the first num_chars number of
    characters in a word
    '''
    def __init__(self, tokenizer, num_chars=5):
        '''TODO
        :param tokenizer: an object with a tokenize(str) method
        :param num_chars: int, word initial characters to keep, defaults to 5
        '''
        self.tokenizer = tokenizer
        self.num_chars = num_chars

