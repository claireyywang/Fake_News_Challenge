
import util
from stanfordcorenlp import StanfordCoreNLP
from collections import Counter
import json
from pprint import pprint
from nltk.corpus import stopwords


from sklearn.feature_extraction.text import CountVectorizer

nlp = StanfordCoreNLP('stanford-corenlp-full-2018-02-27/', lang='en')

tr_stances_file = '../dataset/train_stances.csv'
tr_bodies_file = '../dataset/train_bodies.csv'
dev_stances_file = '../dataset/dev_stances.csv'
dev_bodies_file = '../dataset/dev_bodies.csv'
test_stances_file = '../dataset/test_stances.csv'
test_bodies_file = '../dataset/test_bodies.csv'

dev_pred_file = 'output/dev_predictions.csv'
test_pred_file = 'output/test_predictions.csv'

train_data = util.FNCData(tr_stances_file, tr_bodies_file)
dev_data = util.FNCData(dev_stances_file, dev_bodies_file)
test_data = util.FNCData(test_stances_file, test_bodies_file)


nlp_props = {
    'annotators': 'tokenize,lemma',
    'pipelineLanguage': 'en',
    'outputFormat': 'json'
}

max_n = 6
stopwords = stopwords.words('english')


all_ngrams = {i : Counter() for i in range(1, max_n + 1)}

# TODO: do we want to convert the original ngrams to strings
def ngrams_from_tokens(tokens, max_n):
    # map from n -> ngrams
    output = {}
    
    # remove stopwords from ngrams when n=1
    one_grams = Counter(tokens)
    for w in stopwords:
        one_grams[w] = 0
    output[1] = one_grams
    
    for n in range(2, max_n + 1):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i + n]))
        output[n] = Counter(ngrams)
    
    return output

def lemmatize_text(text):
    nlp_result = json.loads(nlp.annotate(text, properties=nlp_props))

    lemmatized_tokens = []
    for sentance in nlp_result['sentences']:
        for token in sentance['tokens']:
            lemmatized_tokens.append(token['lemma'])

    return lemmatized_tokens

body_tfs = {}


pprint(stopwords.words('english'))

# mapping from body mapping to X array index

for body_id, body_text in train_data.bodies.items():
    body_tokens = lemmatize_text(body_text)
    

    pprint(body_tokens)
    tfs = ngrams_from_tokens(body_tokens, 6)
    body_tfs[body_id] = tfs


# for instance in train_data.instances:
#     body_text = train_data.bodies[instance['Body ID']]
#     header_text = instance['Headline']

    # header_tokens = lemmatize_text(header_text)
    # pprint(ngrams_from_tokens(header_tokens, 6))
    # pprint(all_ngrams)


# print(body_id)
# print(train_data.bodies[body_id])


