### Instructions

To begin, ensure that the latest versions of <code>numpy</code> and <code>tensorflow</code> have been installed.

Navigate to the <code>milestone_3</code> folder and run <code>python3 mlp.py</code>. This will load the FNC data, extract the appropriate features, build and train a neural net, and predict stances for both the dev and test sets. These predictions will be written to <code>output/dev_predictions.csv</code> and <code>output/test_predictions.csv</code> respectively.

Once these stances have been generated, you can evaluate the performance of a particular set of predictions using the evaluation script found in <code>mileston_3/evaluate.py</code>, which is used as follows: <code>python3 evaluate.py gold_stances.csv predictions.csv</code>. For example, if we wanted to evaluate our dev set predictions, we would run <code>python3 evaluate.py ../dataset/dev_stances.csv output/dev_predictions.csv</code>. This will print a confusion matrix of the various actual stances vs. predicted stances, and will output a score (and percentage) based on the FNC's weighted evaluation metric.

### Current System
With training for 90 epochs (as suggested in the paper), we achieved the following scores for our system:

##### Test

Score: 3717.75 out of 5741.25 (64.75506205094709%)
Confusion Matrix:
|           | agree | disagree | discuss | unrelated |
|-----------|-------|----------|---------|-----------|
| agree     | 436   | 53       | 433     | 37        |
| disagree  | 114   | 40       | 125     | 41        |
| discuss   | 552   | 194      | 1364    | 94        |
| unrelated | 1451  | 889      | 653     | 6040      |

##### Dev

Score: 3824.0 out of 5910.0	(64.70389170896784%)
Confusion Matrix:
|           | agree | disagree | discuss | unrelated |
|-----------|-------|----------|---------|-----------|
| agree     | 455   | 45       | 403     | 41        |
| disagree  | 145   | 37       | 160     | 35        |
| discuss   | 571   | 198      | 1414    | 77        |
| unrelated | 1524  | 892      | 750     | 6150      |

### Random Baseline

Our random baseline was a majority class classifier that simply marked all headlines as "unrelated". We achieved the following scores:

##### Test

Score: 2258.25 out of 5741.25 (39.33376877857609%)
Confusion Matrix:
|           | agree | disagree | discuss | unrelated |
|-----------|-------|----------|---------|-----------|
| agree     | 0     | 0        | 0       | 959       |
| disagree  | 0     | 0        | 0       | 320       |
| discuss   | 0     | 0        | 0       | 2204      |
| unrelated | 0     | 0        | 0       | 9033      |

##### Dev

Score: 2329.0 out of 5910.0	(39.407783417935704%)
Confusion Matrix:
|           | agree | disagree | discuss | unrelated |
|-----------|-------|----------|---------|-----------|
| agree     | 0     | 0        | 0       | 944       |
| disagree  | 0     | 0        | 0       | 377       |
| discuss   | 0     | 0        | 0       | 2260      |
| unrelated | 0     | 0        | 0       | 9316      |

Clearly, our new baseline performs significantly better than the random baseline--the scores on both the test and dev sets were over 25% higher than the score achieved by the majority class classifier.