# coding=utf-8

from pathlib import Path
import pytest

import topic_modeling.preprocessing

TEST_FILE_DIR = Path(__file__).resolve().parent / 'test_files'

TEST_DOC_UTF8 = TEST_FILE_DIR / 'Tolstoy_AnnaKareninaSnippet.txt'

TEST_DOC_WINDOWS1251 = TEST_FILE_DIR /'Tolstoy_AnnaKarenina_Windows1251.txt'

def test_blank_lines_non_utf8_encoding_detection():
    with pytest.raises(UnicodeDecodeError):
        topic_modeling.preprocessing.split_doc_on_blank_lines(TEST_DOC_WINDOWS1251)


def test_word_count_non_utf8_encoding_detection():
    with pytest.raises(UnicodeDecodeError):
        topic_modeling.preprocessing.split_doc_on_word_count(TEST_DOC_WINDOWS1251)


def test_line_count_non_utf8_encoding_detection():
    with pytest.raises(UnicodeDecodeError):
        topic_modeling.preprocessing.split_doc_on_line_count(TEST_DOC_WINDOWS1251)


def test_line_break_splitting():
    result = topic_modeling.preprocessing.split_doc(
        TEST_DOC_UTF8, 'line_break')
    assert len(result) == 2
    assert result[0][0] == 'Tolstoy_AnnaKareninaSnippet_0'
    assert result[1][0] == 'Tolstoy_AnnaKareninaSnippet_1'
    assert set([l[1] for l in result]) == set(['Tolstoy'])
    assert result[0][2].startswith('Все счастливые семьи похожи друг на друга, каждая несчастливая семья несчастлива по-своему. Все смешалось в доме Облонских.')
    assert result[0][2].endswith('повар ушел еще вчера со двора, во время обеда; черная кухарка и кучер просили расчета.')
    assert result[1][2].startswith('На третий день после ссоры князь Степан Аркадьич Облонский --')
    assert result[1][2].endswith('но вдруг вскочил, сел на диван и открыл глаза.')

def test_word_type_splitting():
    result = topic_modeling.preprocessing.split_doc(
        TEST_DOC_UTF8, 'word_count', word_count=50)
    assert len(result) == 5
    assert result[0][0] == 'Tolstoy_AnnaKareninaSnippet_0'
    assert result[4][0]  == 'Tolstoy_AnnaKareninaSnippet_4'
    assert set([l[1] for l in result]) == set(['Tolstoy'])
    assert result[0][2].startswith('Все счастливые семьи похожи друг на друга,')
    assert result[0][2].endswith('Положение это продолжалось уже третий день и мучительно')
    assert result[3][2].startswith('как его звали в свете,')
    assert result[3][2].endswith('репко обнял подушку и прижался')
    assert result[4][2] == 'к ней щекой; но вдруг вскочил, сел на диван и открыл глаза.'

def test_line_count_splitting():
    result = topic_modeling.preprocessing.split_doc(
        TEST_DOC_UTF8, 'line_count', line_count=2, word_count=100)

    assert len(result) == 2
    assert result[0][0] == 'Tolstoy_AnnaKareninaSnippet_0'
    assert result[1][0]  == 'Tolstoy_AnnaKareninaSnippet_1'
    assert set([l[1] for l in result]) == set(['Tolstoy'])
    assert result[0][2].startswith('Все счастливые семьи похожи друг на друга,')
    assert result[0][2].endswith('черная кухарка и кучер просили расчета.')
    assert result[1][2].startswith('На третий день после')
    assert result[1][2].endswith('сел на диван и открыл глаза.')


