from sklearn.svm import LinearSVC

from readfile import read_file
from features_extraction import *


tr_stances_file = 'dataset/train_stances.csv'
tr_bodies_file = 'dataset/train_bodies.csv'
dev_stances_file = 'dataset/dev_stances.csv'
dev_bodies_file = 'dataset/dev_bodies.csv'
test_stances_file = 'dataset/test_stances.csv'
test_bodies_file = 'dataset/test_bodies.csv'

print('loading data file....')
tr_stances, tr_articles = read_file(tr_stances_file, tr_bodies_file)
dev_stances, dev_articles = read_file(dev_stances_file, dev_bodies_file)
test_stances, test_articles= read_file(test_stances_file, test_bodies_file)
print("Number of train, dev, test stances: " +
      str(len(tr_stances))+', '+str(len(dev_stances))+', '+str(len(test_stances)))
print("Number of train, dev, test articles: " +
      str(len(tr_articles))+', '+str(len(dev_articles))+', '+str(len(test_articles)))

print('extracting features...')
train_X, train_y = generate_features(tr_stances, tr_articles)
dev_X, dev_y = generate_features(dev_stances, dev_articles)
test_X, test_y = generate_features(test_stances, test_articles)

print('training model...')
clf = LinearSVC()
clf.fit(train_X, train_y)

print('predicting dev dataset..')
dev_pred = [LABELS[int(a)] for a in clf.predict(dev_X)] # predicted labels
dev_gt = [LABELS[int(a)] for a in dev_y] # ground truth

print('evaluating model performance...')
# TODO add evaluation metric 
