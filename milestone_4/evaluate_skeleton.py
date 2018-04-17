import sys
from csv import DictReader

## primarily adopted from Fake News Challenge score.py
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    """
    TODO: Iterate through the gold labels and test lables, calculating the appropriate score
          for each prediction based on the metric in the writeup. Also, update the confusion
          matrix to include counts for each true-pred pair.

    """

    return score, cm

def print_confusion_matrix(cm):
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))

def report_score(actual,predicted):
    score,cm = score_submission(actual,predicted)
    best_score, _ = score_submission(actual,actual)

    print_confusion_matrix(cm)
    # your final score is what the score/best score percentage is
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score

def get_stances_from_csv(filename):
    stances = []
    with open(filename, 'r') as f:
        r = DictReader(f)
        for line in r:
            stances.append(line['Stance'])
    
    return stances

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(len(sys.argv))
        print('usage python3 evaluate.py gold.csv pred.csv')
        exit(1)
    
    gold = get_stances_from_csv(sys.argv[1])
    pred = get_stances_from_csv(sys.argv[2])

    report_score(gold, pred)
