# CIS 530 Milestone 3
#### Rani Iyer, Paul Zuo, Sam Akhavan, Graham Mosley, Claire Wang, Anosha Minai

## Research Paper #1 Paul
[**A simple but tough-to-beat baseline for the Fake News Challenge stance detection task**](https://arxiv.org/pdf/1707.03264.pdf) Benjamin Riedel1, Isabelle Augenstein12, Georgios P. Spithourakis1, Sebastian Riedel1

### Summary
This paper outlines a simple approach to identifying public misinformation. It offers an alternative to more complex, ensemble models that were included in other top submissions to the Fake News Challenge. The model that was used was a multi-layer perceptron with one hidden layer, with features consisting of lexical and similarity features. Specifically, features include the TF vector of the headline, TF vector of the body, and the cosine similarity between the TF-IDF vectors of the headline and the body. Then, the MLP classifier has one hidden layer of 100 units and a softmax on the output of the final linear layer, predicting with the highest scoring label. The objective was to minimize the cross entropy between the model’s softmax probabilities and the true labels- L2 regularization was used for the weights, and dropout was applied on the output of both perceptron layers. 

  

The paper achieved a FNC-1 score of 81.72%, with 97.90% accuracy on unrelated, 81.38% accuracy on discuss, 6.6% accuracy on disagree, and 44.04% accuracy on agree. The real advantage to this approach is that its relative simplicity allows us to understand how it works, what contributes to its performance, and what its limitations are.


## Research Paper #2 ==Rani==
[**Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media**](https://arxiv.org/pdf/1610.09786.pdf) Abhijnan Chakraborty, Bhargavi Paranjape, Sourya Kakarla, Niloy Ganguly

### Summary
The paper assessed the ability to computationally assess clickbait across the web and then built a browser extension to warn readers of clickbait sites. They found their clickbait article data by scraping well known "clickbait" sources, like *Buzzfeed*, *Upworthy*, and *ViralNova*. They then manually categorized these headlines as clickbait/non-clickbait by three reviewers each. For non-clickbait articles, they collected Wikinews articles, which are publically verified by the community before being published. All their data were English headlines.

This paper used several methods of feature extraction and an SVM prediction model, acheiving 93% accuracy in detecting clickbaits. Their features took four types: Sentence Structure, word Patterns, Clickbait Language, and N-grams. With the Sentence structure, they found that the length of headlines, average word length in headlines,  the ratio of stop words to content words, and the longest separation between syntactically dependent words in a headline, all were relevant features. For word patterns, they used the presence of numbers at the start of the headline, unusual punctuation patterns, and number of contractions as features. To measure clickbait language, their features measured the presence of hyperbole, common clickbait phrases, and internet slang. The N-Gram features were Word, POS, and Syntactic N-Grams. In total, these are 14 features. After trying SVM, RBF, Decision Trees, and Random forest classifiers, the authors chose the SVM classifier as their final. With this classifier and feature set, they found 93% accuracy, well above the 78% accuracy of existing classifiers.

## Research Paper #3 Claire 
**[From Clickbait to Fake News Detection: An Approach based on Detecting the Stance of Headlines to Articles](http://aclweb.org/anthology/W17-4215)** by Peter Bourgonje, Julian Moreno Schneider, Georg Rehm

### Summary 
This paper discusses an approach on detecting the stance of headlines with two procedures: the first procedure uses lemmatized n-gram matching for binary classification 'related' or 'unrelated' headline/article pairs; the second procedure uses logistic Regresson to carry out a more fine-grained classification within the 'related' pairs, to further classify them into 'agree', 'disagree' and 'discuss'. 

The first procedure classifies headline/article pairs as 'related' if a resulting score calculated by multiplying the number of matching n-grams and length and IDF value of matching n-gram, then dividing by total number of n-grams, is above certain threshold.  The second procedure uses Mallet's logistic Regression trained on headlines only using the three classes ('agree', 'disagree' and 'discuss'). For each instance, the paper only assigns it to best scoreing class if the distance between the best and second best is above certain threshold. If below the threshold, it uses three binary classifiers to further decide which class to assign to. The method introduced by this paper is different from most other approaches in hhe competition, as they took a more traditional way to approach the problem. The final weighted accuracy score is around 89%, and they are ranked #9 on the leaderboard. 


## Research Paper #4 Graham
**[Team Athene on the Fake News Challenge](https://medium.com/@andre134679/team-athene-on-the-fake-news-challenge-28a5cf5e017b
)** by
Iryna Gurevych Benjamin Schiller, Felix Caspelherr, Avinesh P.V.S. and Debanjan Chaudhuri.

### Summary
This blog post discusses Team Athene's second place model for the Fake News Challenge. The team first experimented with SVM and XGBoost classifiers, but were unable to score much higher than the provided baseline, so they decided to use neural networks. The team used a multilayer perceptron model with 7 hidden layers. Features from both the article heading and body we concatenated to create the full feature vector. Features were created using a unigram bag of words, non-negative matrix factorization, latent semantic indexing, latent semantic analsis and PPDB.

The final model had an accuracy of 89.5% with 99.3% accuracy classifying unrelated articles but only a 9.5% classifying the disagree stance. The authors believe that bidirection LSTMS or CNNs didn't perform well because of the relatively small (~1600 article bodies) size of the training data.



## Research Paper #5
**[Talos Targets Disinformation with Fake News Challenge Victory](http://blog.talosintelligence.com/2017/06/talos-fake-news-challenge.html)** by Talos Intelligence group at Cisco - written by Sean Baird, Doug Sibley, Yuxi Pan

This blog post describes the approach taken by the team that won the Fake News Challenge - the Talos Intelligence research team at Cisco. They used an ensemble approach with two models: a gradient-boosted decision tree model and a deep-learning network. Each model output a % likelihood of each of the four classifications, and each model's predictions were weighted 50% to determine the final prediction.

The deep-learning network was a 1-D CNN (Convolutional Neural Network) applied on the headline and body text, which were represented using Google's word2vec vectors. The output was sent to a multi-class MLP (multi-layer perceptron). A dropout of 0.5 was used in all layers. The GBDT model used features like the number of overlapping words (and 2-grams and 3-grams) between the headline and body text, the TF-IDF and SVD values of these counts, and the presence of some basic sentiment words. The team used XGBoost for their GBDT implementation. The team's final score was 9556.500.


## Our Selected Published Baseline
Though there are some advantages to classic supervised learning approaches like support vector machines, gradient boosting trees, and logistic regression, the vast majority of the top submissions to the Fake News Challenge used a Neural Network classifier. Advantages of classic supervised learning approaches include simplicity of debugging and parameter tuning. Ultimately, our objective in this project is to build a model that has the highest possible accuracy, in which case the neural network models have proven success. 

To combat issues associated with neural network model complexity, we will implement the neural network model discussed in research paper #1. For one, it helps address issues of complexity by having only one hidden layer instead of multiple hidden layers. Secondly, it doesn't involve an ensemble of many different models in addition to the neural network model. Thirdly, it involves additional model considerations that help reduce the risk of overfitting and over complexity, such as L2 regularization. We will follow the guidelines that the paper mentions for the multi-layer perceptron model, including the number of epochs as well as the learn rate and batch size. We will also follow a similar procedure for feature engineering, taking TF vectors of headlines and bodies as well as cosine similarity of the TF-IDF vectors for headline and body.

## Work Cited
Riedel, Benjamin, et al. "A simple but tough-to-beat baseline for the Fake News Challenge stance detection task." _arXiv preprint arXiv:1707.03264_ (2017).

Baird, Sean, et al. "Talos Targets Disinformation with Fake News Challenge Victory." *Talos Intelligence Blog*, 20 June 2017, [link](http://blog.talosintelligence.com/2017/06/talos-fake-news-challenge.html)

Bourgonje, Schneider, Rehm, et al. "From Clickbait to Fake News Detection: An Approach based on Detecting the Stance of Headlines to Articles.", Octavian Popescu, Carlo Strapparava (eds.): _Proceedings of Natural Language Processing meets Journalism, Copenhagen, Denmark, Association for Computational Linguistics, 2017_ 
