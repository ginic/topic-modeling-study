# coding=utf-8
"""Tests for topic_modling.stemming
No lemmatizer will be perfect, but check that each gets at least
half of the lemmas right, just to make sure nothing's totally off the rails.
"""
import copy
import pandas as pd

import topic_modeling.stemming

# This is long, but has a lot of case changes and past tense verbs, should
# give a good indication of each lemmatizer's capabilities
BULGAKOV_TEST_MULTISENTENCE = ("Однажды весною, в час небывало жаркого заката, "
"в Москве, на Патриарших прудах, появились два гражданина. Первый из них, "
"одетый в летнюю серенькую пару, был маленького роста, упитан, лыс, свою "
"приличную шляпу пирожком нес в руке")


BULGAKOV_EXPECTED_LEMMAS = [('однажды', 'однажды'), ('весною', 'весна'), ('в','в'),
    ('час', 'час'), ('небывало', 'небывалый'), ('жаркого', 'жаркий'),
    ('заката', 'закат'), ('в', 'в'), ('Москве', 'Москва'), ('на', 'на'),
    ('Патриарших', 'Патриарший'), ('прудах', 'пруд'), ('появились', 'появиться'),
    ('два', 'два'), ('гражданина', 'гражданин'), ('Первый', 'первый'), ('из', 'из'),
    ('них', 'они'), ('одетый', 'одетый'), ('в', 'в'), ('летнюю','летний'),
    ('серенькую', 'серенький'), ('пару', 'пара'), ('был', 'быть'),
    ('маленького', 'маленький'), ('роста', 'рост'), ('упитан', 'упитанный'),
    ('лыс', 'лысый'), ('свою', 'свой'), ('приличную', 'приличный'),
    ('шляпу', 'шляпа'), ('пирожком', 'пирожок'), ('нес', 'нести'), ('в', 'в'),
    ('руке', 'рука')]

GERMAN_TEST_SENTENCE = """Wikipedia ist ein Projekt zum Aufbau einer Enzyklopädie aus freien Inhalten, zu denen du sehr gern beitragen kannst."""

GERMAN_EXPECTED_LEMMAS =  [('Wikipedia','Wikipedia'), ('ist','sein'), ('ein','einen'),
    ('Projekt','Projekt'), ('zu','zum'), ('Aufbau','Aufbau'), ('einer','einer'),
    ('Enzyklopädie','Enzyklopädie'), ('aus','aus'), ('freien','frei'),
    ('Inhalten','Inhalt'), ('zu','zu'), ('denen','der'), ('du','du'), ('sehr','sehr'),
    ('gern','gern'), ('beitragen','beitragen'), ('kannst','können')]


def helper_test_lemmatizer(lemmatizer, text, expected_lemmas, single_word, single_lemma):
    """These lemmatization tests are all executed the same way. Let them pass
    if at least half the lemmas are correct. We want to make sure the
    dictionaries install correctly and that results are in the correct format.

    :param lemmatizer: Object with lemmatize(str) function
    """
    lemma_pairs = lemmatizer.lemmatize(text)
    assert len(lemma_pairs) == len(expected_lemmas)
    assert lemmatizer.single_term_lemma(single_word) == single_lemma

    # Check correct lemmas by removing them
    expected_copy = copy.deepcopy(expected_lemmas)
    for p in lemma_pairs:
        if p in expected_copy:
            expected_copy.remove(p)
    # pass if at least half of lemmas are right
    assert len(expected_copy) <= len(expected_lemmas)/2


def test_stanza():
    """Test StanzaLemmatizer"""
    lemmatizer = topic_modeling.stemming.StanzaLemmatizer()
    helper_test_lemmatizer(lemmatizer, BULGAKOV_TEST_MULTISENTENCE, BULGAKOV_EXPECTED_LEMMAS, 'руке', 'рука')


def test_snowball():
    """Test SnowballStemmer"""
    stemmer = topic_modeling.stemming.SnowballStemmer()
    # Snowball is more agressive and actually stems, instead of lemmatizes
    expected = [('Однажды', 'однажд'), ('весною', 'весн'), ('в', 'в'),
        ('час', 'час'), ('небывало', 'небыва'), ('жаркого', 'жарк'),
        ('заката', 'закат'), ('в', 'в'), ('Москве', 'москв'), ('на', 'на'),
        ('Патриарших', 'патриарш'), ('прудах', 'пруд'), ('появились', 'появ'),
        ('два', 'два'), ('гражданина', 'гражданин'), ('Первый', 'перв'),
        ('из', 'из'), ('них', 'них'), ('одетый', 'одет'), ('в', 'в'),
        ('летнюю','летн'), ('серенькую', 'сереньк'), ('пару', 'пар'),
        ('был', 'был'), ('маленького', 'маленьк'), ('роста', 'рост'),
        ('упитан', 'упита'), ('лыс', 'лыс'), ('свою', 'сво'),
        ('приличную', 'приличн'), ('шляпу', 'шляп'), ('пирожком', 'пирожк'),
        ('нес', 'нес'), ('в', 'в'), ('руке', 'рук')]
    assert expected == stemmer.lemmatize(BULGAKOV_TEST_MULTISENTENCE)

    assert stemmer.single_term_lemma('руке') == 'рук'


def test_pymystem3():
    """Test Pymystem3Lemmatizer"""
    lemmatizer = topic_modeling.stemming.Pymystem3Lemmatizer()
    helper_test_lemmatizer(lemmatizer, BULGAKOV_TEST_MULTISENTENCE, BULGAKOV_EXPECTED_LEMMAS, 'руке', 'рука')


def test_pymorphy2():
    """Test Pymorphy2Lemmatizer
    """
    lemmatizer  = topic_modeling.stemming.Pymorphy2Lemmatizer()
    helper_test_lemmatizer(lemmatizer, BULGAKOV_TEST_MULTISENTENCE, BULGAKOV_EXPECTED_LEMMAS, 'руке', 'рука')


def test_truncation():
    """Test TruncationStemmer"""
    lemmatizer = topic_modeling.stemming.TruncationStemmer(num_chars=6)
    expected = ([('Однажды', 'Однажд'), ('весною', 'весною'), ('в','в'),
        ('час', 'час'), ('небывало', 'небыва'), ('жаркого', 'жарког'),
        ('заката', 'заката'), ('в', 'в'), ('Москве', 'Москве'), ('на', 'на'),
        ('Патриарших', 'Патриа'), ('прудах', 'прудах'), ('появились', 'появил'),
        ('два', 'два'), ('гражданина', 'гражда'), ('Первый', 'Первый'), ('из', 'из'),
        ('них', 'них'), ('одетый', 'одетый'), ('в', 'в'), ('летнюю','летнюю'),
        ('серенькую', 'серень'), ('пару', 'пару'), ('был', 'был'),
        ('маленького', 'малень'), ('роста', 'роста'), ('упитан', 'упитан'),
        ('лыс', 'лыс'), ('свою', 'свою'), ('приличную', 'прилич'),
        ('шляпу', 'шляпу'), ('пирожком', 'пирожк'), ('нес', 'нес'), ('в', 'в'),
        ('руке', 'руке')])
    assert expected == lemmatizer.lemmatize(BULGAKOV_TEST_MULTISENTENCE)
    assert lemmatizer.single_term_lemma('руке') == 'руке'


def test_stem_counter_update():
    stem_counter = topic_modeling.stemming.ByAuthorStemCounts()
    stem_counter.update('Bulgakov', BULGAKOV_EXPECTED_LEMMAS)
    count_pair = topic_modeling.stemming.NormalizedToken('жаркого', 'жаркий')
    stem_counter.update('Bulgakov', [count_pair])
    assert stem_counter.author_map['Bulgakov'][count_pair] == 2


def test_stem_counter_to_df():
    stem_counter = topic_modeling.stemming.ByAuthorStemCounts()
    example_lemmas = [
        topic_modeling.stemming.NormalizedToken('жаркого', 'жаркий'),
        topic_modeling.stemming.NormalizedToken('Москве', 'Москва')
    ]
    stem_counter.update('Bulgakov', example_lemmas)
    stem_counter.update('Bulgakov', example_lemmas)
    stem_counter.update('Tolstoy', example_lemmas)
    result_df = stem_counter.to_dataframe()
    expected_df = pd.DataFrame({'author':['Bulgakov', 'Bulgakov', 'Tolstoy', 'Tolstoy'], 'token':['жаркого', 'москве', 'жаркого', 'москве'], 'normalized':['жаркий', 'москва', 'жаркий', 'москва'], "count":[2,2,1,1]})
    assert result_df.equals(expected_df)


def test_spacy_lemmatizer():
    """Test spaCy stemmer for German"""
    lemmatizer = topic_modeling.stemming.SpaCyLemmatizer()
    helper_test_lemmatizer(lemmatizer, GERMAN_TEST_SENTENCE, GERMAN_EXPECTED_LEMMAS, "siehst", "sehen")


def test_stanza_german_lemmatizer():
    """Check Stanza language switching works"""
    lemmatizer = topic_modeling.stemming.StanzaLemmatizer(language='de')

    helper_test_lemmatizer(lemmatizer, GERMAN_TEST_SENTENCE, GERMAN_EXPECTED_LEMMAS, "kannst", "können")