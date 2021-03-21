# coding=utf-8
'''
Tests for topic_modeling.tokenization
'''

import topic_modeling.tokenization as tokenization

GOGOL_VY= ('--  Паничи! паничи! сюды! сюды! -- говорили они со всех сторон. '
'-- Ось бублики, мак овники, вертычки, буханци хороши! ей-богу, хороши! '
'на меду! сама пекла! 01234.2345, example@example.ru')

def test_mallet_pattern():
    t = tokenization.RegexTokenizer(tokenization.MALLET_DEFAULT_TOKENIZATION)
    result = t.tokenize(GOGOL_VY)
    expected = ['Паничи', 'паничи', 'сюды', 'сюды', 'говорили', 'они',
        'всех', 'сторон', 'Ось', 'бублики', 'мак', 'овники', 'вертычки',
        'буханци', 'хороши', 'ей-богу', 'хороши', 'меду', 'сама', 'пекла',
        'example@example.ru']
    assert result==expected

def test_word_type_pattern():
    t = tokenization.RegexTokenizer()
    result = t.tokenize(GOGOL_VY)
    expected = ['Паничи', 'паничи', 'сюды', 'сюды', 'говорили', 'они',
        'со', 'всех', 'сторон', 'Ось', 'бублики', 'мак', 'овники',
        'вертычки', 'буханци', 'хороши', 'ей-богу', 'хороши', 'на', 'меду',
        'сама','пекла', 'example@example.ru']
    assert result==expected

def test_keep_punct_pattern():
    t = tokenization.RegexTokenizer(tokenization.KEEP_PUNCT_TOKENIZATION)
    result = t.tokenize(GOGOL_VY)
    expected = ['-','-', 'Паничи', '!', 'паничи', '!', 'сюды', '!', 'сюды',
        '!', '-', '-', 'говорили', 'они', 'со', 'всех', 'сторон', '.', '-',
        '-', 'Ось', 'бублики', ',', 'мак', 'овники', ',', 'вертычки', ',',
        'буханци', 'хороши', '!', 'ей-богу', ',', 'хороши', '!',
        'на', 'меду', '!', 'сама','пекла', '!', '01234.2345', ',',
        'example@example.ru']
    assert result==expected