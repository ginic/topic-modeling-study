# coding=utf-8
"""
Module for stopwords, mainly a wrapper around spaCy.
"""
from spacy.lang.ru import Russian
from spacy.lang.de import German

def get_stopwords(language):
    """Returns a set of stop words for the specified language

    :param language: language name or two letter code
    """
    lang = language.lower()
    stopwords = {}
    if lang in ['ru', 'russian']:
        stopwords = Russian().Defaults.stop_words
        stopwords.update([
            'без',
            'ведь', 'всегда',
            'где', 'даже',
            'из-за', 'из-под', 'иногда',
            'ли',
            'над', 'назад', 'нечего', 'ни', 'никем', 'никого', 'никому', 'ничего', 'ничем', 'ничему', 'ничто', 'ну',
            'обо', 'очень',
            'под', 'пока', 'после', 'потому', 'почему',
            'со',
            'также', 'там', 'тоже', 'тут',
            'хотя',
            'часто', 'через']
        )
    elif lang in ['de', 'german', 'deutsch']:
        stopwords = German().Defaults.stop_words
    return stopwords