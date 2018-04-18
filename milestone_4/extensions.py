import numpy as np
import re
import nltk

fake_indicators = [
  'bogus',
  'fake',
  'fraudulent',
  'forged',
  'mock',
  'false',
  'phony',
  'despite',
  'fictitious',
  'dubious',
  'sham',
  'fraud',
  'hoax',
  'deny',
  'denies',
  'doubt',
  'doubts'
]
wnl = nltk.WordNetLemmatizer()

def tokenize(s):
  tokens = []
  token_string = " ".join(re.findall(r'\w+', s)).lower()
  for t in nltk.word_tokenize(token_string):
    tokens.append(wnl.lemmatize(t).lower())
  return tokens


def filter_refutation(titles, articles):
  X = []
  for i in range(len(titles)):
    title, _ = titles[i], articles[i]
    title_tokens = tokenize(title)
    features = [1 if word in title_tokens else 0 for word in fake_indicators]
    X.append(features)
  return X

def compute_overlap_frac(titles, articles):
  X = []
  for i in range(len(titles)):
    title, article = titles[i], articles[i]
    title_tokens = tokenize(title)
    article_tokens = tokenize(article)
    features = [len(set(title_tokens).intersection(article_tokens))/float(len(set(title_tokens).union(article_tokens)))]
    X.append(features)
  return X


def compute_polarity(titles, articles):
  X = []
  for i in range(len(titles)):
    title, article = titles[i], articles[i]
    title_tokens = tokenize(title)
    article_tokens = tokenize(article)
    features = []
    title_polarity = sum([t in fake_indicators for t in title_tokens]) % 2
    article_polarity = sum([t in fake_indicators for t in article_tokens]) % 2
    features.append(title_polarity)
    features.append(article_polarity)
    X.append(features)
  return X

def generate_extended_features(instances, FNCdata):
  titles, articles = [], []

  for instance in instances:
    titles.append(instance['Headline'])
    articles.append(FNCdata.bodies[instance['Body ID']])

  X_overlap = compute_overlap_frac(titles, articles)
  X_refuting = filter_refutation(titles, articles)
  X_polarity = compute_polarity(titles, articles)

  return np.c_[X_polarity, X_refuting, X_overlap]
