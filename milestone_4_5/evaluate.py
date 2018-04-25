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

    errors = []

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            # 25% weighting if it's the same stance
            score += 0.25
            # aim for 75% weighting if it's same stance AND it's agree, disagree, or discuss
            if g_stance != 'unrelated':
                score += 0.50
        else:
            print(i, g_stance, t_stance)
            errors.append((i, g_stance, t_stance))
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        # confusion matrix includes counts for each true-pred pair
        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm, errors

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
    score,cm, errors = score_submission(actual,predicted)
    print("SCORING PREDICTIONS COMPLETE")
    best_score, _ , _ = score_submission(actual,actual)

    with open('errors_extension3.txt', 'a') as f:
        for val in errors:
            f.write(val)
            f.write("\n")

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
