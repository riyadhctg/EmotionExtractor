# general imports
import re
import nltk
import logging
import pickle

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


def tokenizer(sentence):
  '''
  tokenizes sentence to word tokens using nltk
  '''
  word_tokens = []
  from nltk.tokenize import word_tokenize
  nltk.download('punkt')
  
  if type(sentence) == str:
    word_tokens = word_tokenize(sentence)
    return word_tokens
  elif type(sentence) == list:
    return sentence
  else:
    logger.error('Cannot tokenize. Please input sentence')
    raise Exception('Cannot tokenize. Please input sentence')

  
def remove_puncs_digits(tokens):
  '''
  removes punctuations and digits using regex
  '''
  cleaned_tokens = []
  for token in tokens:
    clean_token = re.sub('[^A-Za-z]+','', token)
    if len(clean_token) > 0:
      cleaned_tokens.append(clean_token)

  logger.info('remove_puncs_digits: %s', cleaned_tokens)
  return cleaned_tokens


def remove_stopwords(tokens):
  '''
  removes stopwords using nltk
  '''

  from nltk.corpus import stopwords
  nltk.download('stopwords')
  nltk.download('punkt')

  stop_words = set(stopwords.words('english'))
  filtered_sentence = [w for w in tokens if not w.lower() in stop_words]
  filtered_sentence = []
    
  for w in tokens:
      if w not in stop_words:
          filtered_sentence.append(w)
    
  logger.info('remove_stopwords: %s', filtered_sentence)
  return filtered_sentence


def filter_by_pos(tokens, allowed_tags=['RB', 'RBS', 'RBR', 'JJ', 'JJR', 'JJS']):
  '''
  allowed tags can take in tags from this TAGSET list: https://github.com/nltk/nltk/blob/develop/nltk/app/chunkparser_app.py
  a more readable list from third party: https://www.guru99.com/pos-tagging-chunking-nltk.html 
  '''

  from nltk.corpus import wordnet as wn
  from nltk import pos_tag
  nltk.download('averaged_perceptron_tagger')

  sent = pos_tag(tokens)
  filtered = [s[0] for s in sent if s[1] in allowed_tags]

  logger.info('filter_by_pos: %s', filtered)
  return filtered


def extract_emo_words(tokens):
  emo_extract_lex = None
  import os
  file = os.path.join(os.path.dirname(__file__),'data/data.pkl')
  with open(file, 'rb') as f:
      emo_extract_lex = pickle.load(f)
  emo_words = []
  for token in tokens:
    if token in emo_extract_lex:
      emo_words.append(token)
  
  logger.info('extract_emo_words: %s', emo_words)
  return emo_words


def extract_neg_emo_words(tokens):
  emo_extract_lex = get_emo_extract_lex()
  emo_words = []
  for token in tokens:
    if token in emo_extract_lex and emo_extract_lex[token] == 'N':
      emo_words.append(token)
  
  logger.info('extract_neg_emo_words: %s', emo_words)
  return emo_words


def extract_pos_emo_words(tokens):
  emo_extract_lex = get_emo_extract_lex()
  emo_words = []
  for token in tokens:
    if token in emo_extract_lex and emo_extract_lex[token] == 'P':
      emo_words.append(token)
  
  logger.info('extract_pos_emo_words: %s', emo_words)
  return emo_words


def lemmatize_tokens(tokens):
  from nltk.stem import WordNetLemmatizer 
  nltk.download('wordnet')
  nltk.download('omw-1.4')

  lemmatizer = WordNetLemmatizer()
  lemmas = []
  lemma_dict = dict()
  for token in tokens:
    lemmatized = lemmatizer.lemmatize(token)
    lemmas.append(lemmatized)
    lemma_dict.update({lemmatized: token})
  # lemma dict is required to retrieve the original words in the input
  
  logger.info('lemmatize_tokens: %s', lemmas)
  return lemmas, lemma_dict


def run_pipeline(
  s, 
  lemmatize=False, 
  clean_stopwords=True, 
  remove_pos=False,
  only_extract_neg=False,
  only_extract_pos=False,
  allowed_pos=None):

  '''
  runs all the emo extractor internal functions according the params supplied
  '''

  # mandatory steps
  t = tokenizer(s)
  t = remove_puncs_digits(t)

  # optional steps
  if clean_stopwords:
    t = remove_stopwords(t)

  if remove_pos:
    if allowed_pos is None:
      t = filter_by_pos(t)
    else:
      t = filter_by_pos(t, allowed_pos)
  
  if lemmatize:
    t, lemma_dct = lemmatize_tokens(t)

  if only_extract_neg:
    t = extract_neg_emo_words(t)
  elif only_extract_pos:
    t = extract_pos_emo_words(t)
  else:
    t = extract_emo_words(t)

  if lemmatize:
    unlemmatized = []
    for lemmas in t:
      unlemmatized.append(lemma_dct[lemmas])
    return unlemmatized
  
  return t