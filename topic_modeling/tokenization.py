# coding=utf-8
'''Rule based tokenization and punctuation cleaning strategies.
'''
# string.punctuation without hypens for compound words
# «» are added for quotations and – (em dash, not a hypen) added for direct speech
RU_PUNCTUATION='!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~«»–'

def clean_punctuation(text):
    '''Returns a string that matches the input text with punctuation removed

    :param text: str
    '''
    return text.translate(str.maketrans('', '', RU_PUNCTUATION))


class Tokenizer:
    ''' A tokenization strategy that closely matches Mallet.
    :TODO
    '''
    def __init__(self):
        pass