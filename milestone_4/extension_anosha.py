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

#clf = LogisticRegression(penalty='l2') # 77.09122577835838 on test
clf1 = LogisticRegression(penalty='l1') # 77.8314826910516 on test
# clf = GaussianNB() # 41.96385804485086 on test
# clf = BernoulliNB() # 49.87154365338559 on test
clf3 = LinearSVC(C=10, max_iter=100, dual=False) # 0.76 on test
clf2 = RandomForestClassifier(n_estimators = 100, max_features='sqrt') #80.3004572175049 on test
clf4 = RidgeClassifier() # 75.28412802090138 on test
#clf = GradientBoostingClassifier()
clf5 = SGDClassifier()
# clf6 = Perceptron()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3), ('rc', clf4), ('sgd', clf5)], weights=[1, 2, 1, 1, 1], voting='hard')
eclf = eclf.fit(train_X, train_y)

# clf.fit(train_X, train_y)

print('predicting test dataset..')
#test_pred = [LABELS[int(a)] for a in clf.predict(test_X)] # predicted labels
#test_gt = [LABELS[int(a)] for a in test_y] # ground truth
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