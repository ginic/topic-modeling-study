# coding=utf-8
'''Shared functionalities for stemming and lemmatization

# TODO truncation 'stemmer' with specified max character length
# TODO NLTK Russian Porter (snowball) stemmer
# TODO Pymystem (POS tag sensitive)
'''
import stanza

STANZA_SETTINGS='tokenize,lemma'
STANZA_PACKAGE='syntagrus'

class StanzaLemmatizer:
    def __init__(self):
        stanza.download('ru', processors=STANZA_SETTINGS, package=STANZA_PACKAGE)
        self.pipeline = stanza.Pipeline('ru',processors=STANZA_SETTINGS,
                                        package=STANZA_PACKAGE)

    def lemmatize(self, text):
        '''Returns list of (word, lemma) pairs for each word in the given text.
        Stanza's sentence breaking and tokenization is used.

        :param text: str, Russian text to process
        '''
        result = []
        doc = self.pipeline(text)
        for sent in doc.sentences:
            result.extend([(word.text, word.lemma) for word in sent.words])
        return result