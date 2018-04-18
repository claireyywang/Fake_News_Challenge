# Extension 1

## Instructions

This extension implementation used the same file structure as published baseline. To train the model and run the predictions on dev and test data, navigate to `milestone_4` file, and run  `python3 extended_mlp.py`. The predictions on dev and test data will be written to `output` folder. To compute the confusion matrix for performance, run `python3 evaluate.py ../dataset/dev_stances.csv output/dev_predictions.csv`and  `python3 evaluate.py ../dataset/test_stances.csv output/test_predictions.csv`.

## Extension Description

This extension is built on the original structure of multiperceptation neural network model. Instead of just using the TF of headlines, TF of body articles and the cosine of TF-IDF for the headline/article pair, we appended three more features to the feature set: overlapping words, filtered refuting words and word polarity. Overlapping words feature computes the fraction of word tokens used in both headline and body article among all tokens used in headline and article; filtered refuting word feature shows whether a refuting word (among a defined list of refuting words) showed up in the headline; word polarity calculates the number of refuting words in headline or body article. Each of them represents a property of texts appeared in headline and article pairs. The implementation of extension can be found in `extensions.py`.

## Performance

### Dev data

-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |    429    |    60     |    412    |    43     |
-------------------------------------------------------------
| disagree  |    145    |    28     |    167    |    37     |
-------------------------------------------------------------
|  discuss  |    535    |    206    |   1437    |    82     |
-------------------------------------------------------------
| unrelated |   1547    |    862    |    727    |   6180    |
-------------------------------------------------------------
Score: 3820.25 out of 5910.0 
Accuracy: **(64.6404399323181%)**

### Test data

-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |    397    |    57     |    463    |    42     |
-------------------------------------------------------------
| disagree  |    93     |    37     |    140    |    50     |
-------------------------------------------------------------
|  discuss  |    580    |    194    |   1314    |    116    |
-------------------------------------------------------------
| unrelated |   1503    |    791    |    689    |   6050    |
-------------------------------------------------------------
Score: 3642.25 out of 5741.25 
Accuracy: **(63.44001741780971%)**

## Performance Analysis

The performance of this extension did not improve compared to the published baseline performance (64.703% on dev, 64.755% on test), but is within 1% variation. Adding features did not achieve the expected improvement on performance. 
