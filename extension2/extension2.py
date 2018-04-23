
import util
from stanfordcorenlp import StanfordCoreNLP
from collections import Counter
import json
from pprint import pprint

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

all_ngrams = {i : Counter() for i in range(1, 6 + 1)}
print(all_ngrams)

# TODO: do we want to convert the original ngrams to strings
def ngrams_from_tokens(tokens, max_n, is_body):
    output = {}
    output[1] = Counter(tokens)
    for n in range(2, max_n + 1):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i + n]))
        n_tf = Counter(ngrams)
        if is_body:
            for term in n_tf.keys():
                all_ngrams[n][term] += 1
        output[n] = n_tf
    return output

def lemmatize_text(text):
    nlp_result = json.loads(nlp.annotate(text, properties=nlp_props))

    lemmatized_tokens = []
    for sentance in nlp_result['sentences']:
        for token in sentance['tokens']:
            lemmatized_tokens.append(token['lemma'])

    return lemmatized_tokens

# print(len(train_data.bodies))
# print(len(train_data.instances))

body_tfs = {}

for body_id, body_text in train_data.bodies.items():
    body_tokens = lemmatize_text(body_text)
    tfs = ngrams_from_tokens(body_tokens, 6, is_body=True)
    body_tfs[body_id] = tfs

# print(all_ngrams)

pprint(all_ngrams[2].most_common(5))


# idfs = {}

# def get_idf(s, n):



# for instance in train_data.instances:
#     body_text = train_data.bodies[instance['Body ID']]
#     header_text = instance['Headline']
    
#     header_tokens = lemmatize_text(header_text)

    
#     pprint(ngrams_from_tokens(header_tokens, 6))

#     pprint(all_ngrams)

#     exit()


    
    # 
    # pprint()
    
    # exit()


# print(body_id)
# print(train_data.bodies[body_id])


