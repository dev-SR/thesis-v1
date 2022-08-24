import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()

REPLACE_PUNCTUATION_BY_SPACE_RE = re.compile(
    '[/(){}\[\]\|@,:;]')  # exclude full stop
REMOVE_NUM = re.compile('[\d+]')
EMAIL_RE = re.compile('\b[\w\-.]+?@\w+?\.\w{2,4}\b')
PHONE_RE = re.compile(
    '\b(\+\d{1,2}\s?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})\b')
NUMBER_RE = re.compile('\d+(\.\d+)?')
URLS_RE = re.compile('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)')
EXTRA_SPACE_RE = re.compile('\s+')
STOPWORDS = set(stopwords.words('english'))
# BAD_SYMBOLS: except numbers,character,space,newline,.,_
BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z _.\n]')
HYPHENATED_RE = re.compile('-\n')
SPACE_BEFORE_FULLSTOPS = re.compile('\s\.')
MORE_THAN_ONE_FULLSTOPS_RE = re.compile('\.\.+')


def cleanText(text):
    text = HYPHENATED_RE.sub('', text)    # joining Hyphenated words
    text = REPLACE_PUNCTUATION_BY_SPACE_RE.sub(' ', text)
    text = NUMBER_RE.sub('', text)
    text = EMAIL_RE.sub('', text)
    text = URLS_RE.sub('', text)
    text = PHONE_RE.sub('', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = EXTRA_SPACE_RE.sub(' ', text)

    # Lower case
    text = text.lower()

    # Tokenize
    words = word_tokenize(text)

    # Remove Stop Words
    # words = [w for w in words if not w in STOPWORDS]

    # # Stemming
    stemmed_words = [ps.stem(w) for w in words]

    # # Join the words back into one string separated by space,
    stemmed_sen = ' '.join(stemmed_words)
    # removing space before full stops
    stemmed_sen = SPACE_BEFORE_FULLSTOPS.sub('.', stemmed_sen)
    # removing more than one full stop
    stemmed_sen = MORE_THAN_ONE_FULLSTOPS_RE.sub('', stemmed_sen)
    return stemmed_sen
