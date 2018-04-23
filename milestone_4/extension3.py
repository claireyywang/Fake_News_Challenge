from extended_setup import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

tr_stances_file = '../dataset/train_stances.csv'
tr_bodies_file = '../dataset/train_bodies.csv'
dev_stances_file = '../dataset/dev_stances.csv'
dev_bodies_file = '../dataset/dev_bodies.csv'
test_stances_file = '../dataset/test_stances.csv'
test_bodies_file = '../dataset/test_bodies.csv'

dev_pred_file = 'output/dev_predictions.csv'
test_pred_file = 'output/test_predictions.csv'

print('loading data...')
train_data = FNCData(tr_stances_file, tr_bodies_file)
dev_data = FNCData(dev_stances_file, dev_bodies_file)
test_data = FNCData(test_stances_file, test_bodies_file)

bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = create_vectors(train_data, dev_data, test_data, lim_unigram=5000)

# extract features and labels
print('extracting features...')
train_X, train_y = pipeline_train(train_data, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
dev_X, dev_y = pipeline_dev(dev_data, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
test_X = pipeline_test(test_data, bow_vectorizer,tfreq_vectorizer, tfidf_vectorizer)

print('training  data...')

'''
Idea is that the features used for classifying an article as unrelated/related
could be very different than features used for agree/disagree/discuss so we 
first train a classifier to determine if an article is unrelated or related
we then train a second classifier using only the examples that are related
'''

stance_label = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
stance_label_rev = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}

stage_one_map = {0: 1, 1: 1, 2:1, 3:0}
stage_one_map_rev = {0: 'unrelated', 1: 'related'}

# convert labels for stage one
def convert_to_stage_one_fn(original_label):
    return stage_one_map[original_label]

convert_to_stage_one = np.vectorize(convert_to_stage_one_fn)

train_y_stage_one = convert_to_stage_one(train_y)
dev_y_stage_one = convert_to_stage_one(dev_y)

clf_stage_one = LogisticRegression()
clf_stage_one.fit(train_X, train_y)

print('predicting...')
dev_y_pred_stage_one = clf_stage_one.predict(dev_X)
print('stage 1 f1 score {}'.format(f1_score(dev_y_stage_one, dev_y_pred_stage_one)))



