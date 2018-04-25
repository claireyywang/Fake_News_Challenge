import random

# import all functions in setup module
from extended_setup_two import *
from evaluate import score_submission, report_score, get_stances_from_csv
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

tr_stances_file = '../dataset/train_stances.csv'
tr_bodies_file = '../dataset/train_bodies.csv'
dev_stances_file = '../dataset/dev_stances.csv'
dev_bodies_file = '../dataset/dev_bodies.csv'
test_stances_file = '../dataset/test_stances.csv'
test_bodies_file = '../dataset/test_bodies.csv'

dev_pred_file = 'output/dev_predictions.csv'
test_pred_file = 'output/test_predictions.csv'

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

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
test_y = get_stances_from_csv(test_stances_file)

# extract features and labels
print('building model...')

clf1 = LogisticRegression(penalty='l1')
clf2 = RandomForestClassifier(n_estimators = 100, max_features='sqrt')
clf3 = LinearSVC(C=10, max_iter=100, dual=False)
clf4 = RidgeClassifier()
clf5 = SGDClassifier()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3), ('rc', clf4), ('sgd', clf5)], weights=[1, 2, 1, 1, 1], voting='hard')
eclf = eclf.fit(train_X, train_y)

print('predicting test dataset..')
test_pred = eclf.predict(test_X) # predicted labels
test_gt = test_y # ground truth

print('predicting dev dataset..')
dev_pred = eclf.predict(dev_X) # predicted labels
dev_gt = dev_y # ground truth

print('evaluating model performance...')
print('===============================')
print('Test Dataset Performance:')
test_res = report_score(test_gt,test_pred)
print('===============================')
print('Dev Dataset Performance:')
dev_res = report_score(dev_gt, dev_pred)

save_predictions(dev_pred, dev_pred_file)
save_predictions(test_pred, test_pred_file)
