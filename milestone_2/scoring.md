EVALUATION METRIC/SCORING GUIDELINES

First, let's define what the target metric is. What we're trying to predict is the relationship an article's headline has with its body of text- these relationships can be unrelated, agrees, disagrees or discusses. The latter 3 categories fall under the broader category of related. 

We then come up with a weighted evaluation method that puts higher emphasis on correctly classifying related headline-body pairs. If the prediction is unrelated and the true value is unrelated, then the score increases by 0.25. If the prediction is correct and the true value is agrees, disagrees, or discusses (it's related), then the score increases by 1.0. We return a metric that is the score from our predictions divided by what the score would be if we ran it using the true values as the predicted values. 

For that reason, a higher evaluation metric is a good thing.

To call the evaluation script, you merely import the functions from the evaluate.py file and you can just call report_score, which returns that evaluation metric. It also prints out a 4x4 confusion matrix, which includes the count of each true-predicted pair.

The inspiration for this metric comes from the Fake News Challenge (http://www.fakenewschallenge.org/). They used this weighted evaluation metric as their competition assessment criteria. 

As for sample input/output:

intput: 
true: [0,0,0,0,1,1,0,3,3]
pred: [0,0,0,0,1,1,2,3,3]

the call: report_score(true, pred)

output:
-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |     4     |     0     |     1     |     0     |
-------------------------------------------------------------
| disagree  |     0     |     2     |     0     |     0     |
-------------------------------------------------------------
|  discuss  |     0     |     0     |     0     |     0     |
-------------------------------------------------------------
| unrelated |     0     |     0     |     0     |     2     |
-------------------------------------------------------------
Score: 6.75 out of 7.5	(90.0%)
