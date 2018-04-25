import sys
import csv
from pathlib import Path
from collections import Counter, defaultdict
import string
import nltk
from wordcloud import WordCloud

if len(sys.argv) != 2:
    print('usage python3 exploratory_analysis.py <dataset folder>')
    exit(1)

def load_dataset(dataset_dir, dataset_name):
    stances = []
    bodies = []
    with open(dataset_dir.joinpath(dataset_name + '_stances.csv'), 'r') as f:
        stances = list(csv.DictReader(f))
    with open(dataset_dir.joinpath(dataset_name + '_bodies.csv'), 'r') as f:
        bodies = list(csv.DictReader(f))
    return stances, bodies

dataset_dir = Path(sys.argv[1])
dataset_names = ['train', 'dev', 'test']
datasets = {name: load_dataset(dataset_dir, name) for name in dataset_names}
labels = ['agree', 'disagree', 'discuss', 'unrelated']

# stance distribution
print('label distributions')
for dataset_name, (stances, _) in datasets.items():
    print(dataset_name)
    stance_counts = Counter(x['Stance'] for x in stances)
    total = sum(stance_counts.values())
    for stance, count in sorted(stance_counts.items()):
        print('{:10} {:>6} ({:.2f}%)'.format(stance, count, count/total * 100))
    print()

stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.update(c for c in string.punctuation)
stopwords.add('’')
stopwords.add('‘')
stopwords.add("'s")
stopwords.add("n't")

# headline words
print('headline frequencies')
label_word_counts = {l : Counter() for l in labels}
for stances, _ in datasets.values():
    for stance in stances:
        headline_tokens = nltk.word_tokenize(stance['Headline'])
        headline_tokens = [t.lower() for t in headline_tokens 
                           if t.lower() not in stopwords]
        label_counter = label_word_counts[stance['Stance']]
        for token in headline_tokens:
            label_counter[token] += 1

for label, words in label_word_counts.items():
    print(label)
    for word, count in words.most_common(20):
        print('{:10} {:>6}'.format(word, count))
    print()

    wc = WordCloud(width=800, height=400).generate_from_frequencies(words)
    wc.to_file(label + '-headlines.png')
