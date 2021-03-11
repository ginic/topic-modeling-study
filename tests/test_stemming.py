# coding=utf-8
'''Tests for topic_modling.stemming
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


def test_stanza():
    lemmatizer = topic_modeling.stemming.StanzaLemmatizer()
    lemma_pairs = lemmatizer.lemmatize(BULGAKOV_TEST_MULTISENTENCE)
    assert len(lemma_pairs) == len(EXPECTED_LEMMAS)

    # Check correct lemmas
    expected_copy = copy.deepcopy(EXPECTED_LEMMAS)
    for p in lemma_pairs:
        if p in expected_copy:
            expected_copy.remove(p)
    # pass if at least half of lemmas are right
    assert len(expected_copy) <= len(EXPECTED_LEMMAS)/2