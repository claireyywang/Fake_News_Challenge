import re
import nltk
import numpy as np

wnl = nltk.WordNetLemmatizer()

refuting_words = [
  'fake',
  'fraud',
  'hoax',
  'false',
  'deny',
  'denies',
  'refute',
  'not',
  'despite',
  'nope',
  'doubt',
  'doubts',
  'bogus',
  'debunk',
  'pranks',
  'retract'
]

def tokenize(s):
  tokens = []
  token_string = " ".join(re.findall(r'\w+', s)).lower()
  for t in nltk.word_tokenize(token_string):
    tokens.append(wnl.lemmatize(t).lower())
  return tokens

def word_overlap_features(headlines, body_articles):
  X = []
  for i in range(len(headlines)):
    headline, article = headlines[i], body_articles[i]
    headline_tokens = tokenize(headline)
    article_tokens = tokenize(article)
    features = [len(set(headline_tokens).intersection(article_tokens))/float(len(set(headline_tokens).union(article_tokens)))]
    X.append(features)
  return X


def refuting_features(headlines, body_articles):
  X = []
  for i in range(len(headlines)):
    headline, _ = headlines[i], body_articles[i]
    headline_tokens = tokenize(headline)
    features = [1 if word in headline_tokens else 0 for word in refuting_words]
    X.append(features)
  return X


def polarity_features(headlines, body_articles):
  X = []
  for i in range(len(headlines)):
    headline, article = headlines[i], body_articles[i]
    headline_tokens = tokenize(headline)
    article_tokens = tokenize(article)
    features = []
    headline_polarity = sum([t in refuting_words for t in headline_tokens]) % 2
    article_polarity = sum([t in refuting_words for t in article_tokens]) % 2
    features.append(headline_polarity)
    features.append(article_polarity)
    X.append(features)
  return X

def generate_extended_features(instances, FNCdata):
  headlines, body_articles = [], []

  for instance in instances:
    headlines.append(instance['Headline'])
    body_articles.append(FNCdata.bodies[instance['Body ID']])

  X_overlap = word_overlap_features(headlines, body_articles)
  X_refuting = refuting_features(headlines, body_articles)
  X_polarity = polarity_features(headlines, body_articles)

  return np.c_[X_polarity, X_refuting, X_overlap]