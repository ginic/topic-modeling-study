# coding=utf-8
'''Rule based tokenization and punctuation cleaning srategies.
'''
# string.punctuation without hypens for compound words
RU_PUNCTUATION='!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'

def clean_punctuation(text):
    '''Returns a string that matches the input text with punctuation removed

    :param text: str
    '''
    return text.translate(str.maketrans('', '', RU_PUNCTUATION))


class Tokenizer:
    ''' A tokenization strategy that closely matches Mallet.
    '''
