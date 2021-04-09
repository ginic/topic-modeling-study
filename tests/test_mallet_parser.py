# coding=utf-8

from pathlib import Path

import topic_modeling.mallet_parser as mallet_parser

TEST_FILE_DIR = Path(__file__).resolve().parent / 'test_files/toy_example'

def test_get_vocab():
    vocab, vocab_index = mallet_parser.get_vocab(TEST_FILE_DIR/'toy_example_pruned_vocab.txt')
    assert len(vocab) == 1014
    assert len(vocab_index) == 1014
    assert vocab_index['i'] == 0
    assert vocab_index["ящик"] == 1013


