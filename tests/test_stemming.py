# coding=utf-8
'''Tests for topic_modling.stemming
No stemmer/lemmatizer will be perfect, but check that each gets at least
half of the lemmas right, just to make sure nothing's totally off the rails.
'''
import copy

import topic_modeling.stemming

# This is long, but has a lot of case changes and past tense verbs, should
# give a good indication of each lemmatizer's capabilities
BULGAKOV_TEST_MULTISENTENCE = ("Однажды весною, в час небывало жаркого заката, "
"в Москве, на Патриарших прудах, появились два гражданина. Первый из них, "
"одетый в летнюю серенькую пару, был маленького роста, упитан, лыс, свою "
"приличную шляпу пирожком нес в руке")

EXPECTED_LEMMAS = ([('однажды', 'однажды'), ('весною', 'весна'), ('в','в'),
('час', 'час'), ('небывало', 'небывалый'), ('жаркого', 'жаркий'),
('заката', 'закат'), ('в', 'в'), ('Москве', 'Москва'), ('на', 'на'),
('Патриарших', 'Патриарший'), ('прудах', 'пруд'), ('появились', 'появиться'),
('два', 'два'), ('гражданина', 'гражданин'), ('Первый', 'первый'), ('из', 'из'),
('них', 'они'), ('одетый', 'одетый'), ('в', 'в'), ('летнюю','летний'),
('серенькую', 'серенький'), ('пару', 'пара'), ('был', 'быть'),
('маленького', 'маленький'), ('роста', 'рост'), ('упитан', 'упитанный'),
('лыс', 'лысый'), ('свою', 'свой'), ('приличную', 'приличный'),
('шляпу', 'шляпа'), ('пирожком', 'пирожок'), ('нес', 'нести'), ('в', 'в'),
('руке', 'рука')])

def helper_test_lemmatizer(lemmatizer):
    '''These stemming tests are all executed the same way.

    :param lemmatizer: Object with lemmatize(str) function
    '''
    lemma_pairs = lemmatizer.lemmatize(BULGAKOV_TEST_MULTISENTENCE)
    assert len(lemma_pairs) == len(EXPECTED_LEMMAS)

    # Check correct lemmas
    expected_copy = copy.deepcopy(EXPECTED_LEMMAS)
    for p in lemma_pairs:
        if p in expected_copy:
            expected_copy.remove(p)
    # pass if at least half of lemmas are right
    assert len(expected_copy) <= len(EXPECTED_LEMMAS)/2


def test_stanza():
    '''Test StanzaLemmatizer'''
    lemmatizer = topic_modeling.stemming.StanzaLemmatizer()
    helper_test_lemmatizer(lemmatizer)


def test_snowball():
    '''Test SnowballStemmer'''
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


def test_pymystem3():
    '''Test Pymystem3Lemmatizer'''
    lemmatizer = topic_modeling.stemming.Pymystem3Lemmatizer()
    helper_test_lemmatizer(lemmatizer)


def test_truncation():
    '''Test TruncationStemmer'''
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


