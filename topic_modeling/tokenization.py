# coding=utf-8
'''Rule based tokenization and punctuation cleaning strategies.
'''
import regex

# string.punctuation without hypens for compound words
# ₽ for rubles, «» are added for quotations,
# – (em dash, not a hypen) added for direct speech
# Could also use \p{P} with regex replace instead, but I think
# string.translate is faster and I'd like to have a clear list
RU_PUNCTUATION='!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~₽«»–'

# Mallet's default for topic modeling and import
# Note: Doesn't allow for 1 or 2 letter words
MALLET_DEFAULT_TOKENIZATION=r"\p{L}[\p{L}\p{P}]+\p{L}"

# Should allow punctuation between letters, otherwise punctuation is ignored
WORD_TYPE_TOKENIZATION=r"[\p{L}\d]+[\p{P}\p{L}\d]+[\p{L}\d]|[\p{L}\d]+"

# Punctuation not in the middle of a word is kept in its own token
KEEP_PUNCT_TOKENIZATION=r"[\p{L}\d]+[\p{P}\p{L}\d]+[\p{L}\d]|[\p{L}\d]+|\p{P}"

def clean_punctuation(text):
    '''Returns a string that matches the input text with punctuation removed

    :param text: str
    '''
    return text.translate(str.maketrans('', '', RU_PUNCTUATION))


class RegexTokenizer:
    '''Tokenizer relying on the regex library for compiling expressions.
    '''
    def __init__(self, pattern=WORD_TYPE_TOKENIZATION):
        self.pattern = regex.compile(pattern)

    def tokenize(self, text):
        return self.pattern.findall(text)