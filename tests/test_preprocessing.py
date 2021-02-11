# coding=utf-8

from pathlib import Path
import pytest

import topic_modeling.preprocessing

TEST_FILE_DIR = Path(__file__).resolve().parent / 'test_files'

TEST_DOC_UTF8 = TEST_FILE_DIR / 'AnnaKareninaSnippet_Tolstoy.txt'

TEST_DOC_WINDOWS1251 = TEST_FILE_DIR /'AnnaKarenina_Tolstoy_Windows1251.txt'

def test_blank_lines_non_utf8_encoding_detection():
    with pytest.raises(UnicodeDecodeError):
        topic_modeling.preprocessing.split_doc_on_blank_lines(TEST_DOC_WINDOWS1251)


def test_word_count_non_utf8_encoding_detection():
    with pytest.raises(UnicodeDecodeError):
        topic_modeling.preprocessing.split_doc_on_word_count(TEST_DOC_WINDOWS1251)


def test_line_break_splitting():
    result = topic_modeling.preprocessing.split_doc_on_blank_lines(TEST_DOC_UTF8)
    assert len(result) == 2
    assert result[0][0] == 'AnnaKareninaSnippet_Tolstoy_0'
    assert result[1][0] == 'AnnaKareninaSnippet_Tolstoy_1'
    assert set([l[1] for l in result]) == set(['AnnaKareninaSnippet_Tolstoy'])
    assert result[0][2].startswith("Все счастливые семьи похожи друг на друга, каждая несчастливая семья несчастлива по-своему. Все смешалось в доме Облонских.")
    assert result[0][2].endswith("повар ушел еще вчера со двора, во время обеда; черная кухарка и кучер просили расчета.")
    assert result[1][2].startswith("На третий день после ссоры князь Степан Аркадьич Облонский --")
    assert result[1][2].endswith("но вдруг вскочил, сел на диван и открыл глаза.")

def test_100_word_type_splitting():
    pass

