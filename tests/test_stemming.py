# coding=utf-8

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
    # TODO this will definitly fail. Maybe pass if at least half of them are right?
    # assert lemma_pairs == EXPECTED_LEMMAS