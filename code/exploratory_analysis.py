import sys
import csv
from pathlib import Path
from collections import Counter

if len(sys.argv) != 2:
    print('usage python3 exploratory_analysis.py <dataset folder>')

def load_dataset(dataset_dir, dataset_name):
    stances = []
    bodies = []
    with open(dataset_dir.joinpath(dataset_name + '_stances.csv'), 'r') as f:
        stances = list(csv.DictReader(f))
    with open(dataset_dir.joinpath(dataset_name + '_bodies.csv'), 'r') as f:
        bodies = list(csv.DictReader(f))
    return stances, bodies

dataset_dir = Path(sys.argv[1])
datasets = ['train', 'dev', 'test']

for dataset_name in datasets:
    stances, _ = load_dataset(dataset_dir, dataset_name)
    stance_counts = Counter(x['Stance'] for x in stances)
    total = sum(stance_counts.values())
    for stance, count in sorted(stance_counts.items()):
        print('{:10} {:>6} ({:.2f}%)'.format(stance, count, count/total * 100))
    print()