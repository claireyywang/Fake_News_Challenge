Please upload a markdown file that describes your data (name it data.md). It should give an example of the data, describe the file format of the data, give a link to the full data set (if you’re uploading a sample),

Our task is to evaluate the stance that a headline takes on an article. The possible stances are agrees, disagrees, discusses, or is unrelated. Our data is headlines that are paired with article texts and then labeled with one of the above stances.

The data for each split have two csv files. The 'x_bodies.csv' file contains the article body ID and the full article text. The 'x_stances.csv' file contains a headline, a body id to which it is compared, and the crowdsourced-label.

Our training data, dev, and test set each have about 2500 articles. Our training set has about 60,000 labeled headlines (compared to the articles in the train_bodies.csv file). Our dev set and test set have about 8000 labeled headlines (also compared to articles in the corresponding x_bodies.csv file).  

Our data is from the [Fake News Challenge](http://www.fakenewschallenge.org) that we are following for our project. They provided the split training and test data. We kept their training split as our training split, and split their test data equally to produce a dev and test set. 
