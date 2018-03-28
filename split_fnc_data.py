from collections import defaultdict
import csv
import random

body_rows = {}
body_header = None
stance_rows = defaultdict(list)
stance_header = None

body_filenames = ['original_fnc_data/competition_test_bodies.csv']
for body_filename in body_filenames:
    with open(body_filename) as csvfile:
        reader = csv.reader(csvfile)
        body_header = next(reader, None)
        for row in reader:
            body_rows[int(row[0])] = row

stance_filenames = ['original_fnc_data/competition_test_stances.csv']
for stance_filename in stance_filenames:
    with open(stance_filename) as csvfile:
        reader = csv.reader(csvfile)
        stance_header = next(reader, None)
        for row in reader:
            stance_rows[int(row[1])].append(row)

ids = list(body_rows.keys())
random.shuffle(ids)

dev_ids = []
test_ids = []

for i, body_id in enumerate(ids):
    percent = i / len(ids)
    if percent <= .5:
        dev_ids.append(body_id)
    else:
        test_ids.append(body_id)

for name, ids in [('dev', dev_ids), ('test', test_ids)]:
    with open('{}_bodies.csv'.format(name), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(body_header)
        for body_id in sorted(ids):
            writer.writerow(body_rows[body_id])

    all_stances = []
    for body_id in ids:
        all_stances.extend(stance_rows[body_id])
    random.shuffle(all_stances)

    with open('{}_stances.csv'.format(name), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(stance_header)
        for stance in all_stances:
            writer.writerow(stance)

