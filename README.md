# News classification using different models by using Scikit Learn library

This is an implementation to investigate supervised learning algorithms for news classification on a cleaned BBC news datasets using LR, NB, SVM, NNs. It divided into three parts:

* Task 1: Exploratory data analytics on text data
* Task 2: Perform classification models (LR, NB, SVM, NNs) to build article classifiers using the given dataset.
* Task 3: Investigate the impact of multiple hyperparameters. Compare the classification quality across four classification models in terms of F1 measure. Learn to manage overfitting or underfitting situations.

## Datasets

The datasets has two data files: train.csv, test.csv.

Let’s consider two classes of BBC news articles: tech news and entertainment news. After loading the data (.csv file) using pandas library, you will see that each news article is a comma- separated line with three columns: news ID, processed news body, news class. The processed news bodies are tokenized and lower-cased with removal of stop words and special characters. 


## Task 1: Exploratory Data Analytics

* (a) Load the dataset and construct a feature vector for each article in the. Report the number of articles, and the number of extracted features. Show 5 example articles with their extracted features using a dataframe.
* (b) Conduct term frequency analysis and report three plots: (i) top-50 term frequency distribution across the entire dataset, (ii) term frequency distribution for respective class of articles, and (iii) class distribution.

## Task 2: Classification Models Learning

#### (a) Logistic Regression. 
Train a Logistic Regression classifier with L2-regularization. Consider different values of the regularization term λ. Describe the effect of the regularization parameter λ on the outcome in terms of bias and variance. 
* Report the plot generated for specific λ values with training loss on the y-axis versus λ on the x-axis to support my claim.

#### (b) Naive Bayes. 
Train a Naive Bayes classifier using all articles features. 
* (i) Report the top-20 most identifiable words that are most likely to occur in the articles over two classes using the NB classifier;
* (ii) Report the top-20 words that maximize the following quantity 𝑃(𝑋𝑤=1|𝑌=𝑦)/𝑃(𝑋𝑤=1|𝑌≠𝑦). 
Briefly explain which list of words describe the two classes better.

####  (c) SVM. 
Train a SVM classification models on the training dataset. 
* (i) Report the surface plot for the soft-margin linear SVM with my choice of misclassification penalty (𝐶);
* (ii) Report the surface plot for the hard-margin RBF kernel with my choice of kernel width (σ). 
Explain the impact of penalty 𝐶 on the soft-margin decision boundaries, as well as the kernel hyperparameter on the hard-margin decision boundaries. 

#### (d) Neural Network. 
Consider the Neural Network with the following hyperparameters: the initial weights uniformly drawn in range [0,0.1] with learning rate 0.01. 
* (i) Train a single hidden layer neural network using the hyperparameters on the training dataset, except for the number of hidden units (x) which should vary among 5, 20, and 40. Run the optimization for 100 epochs each time. Namely, the input layer consists of n features x = [x1, ..., xn]T , the hidden layer has x nodes z = [z1, ..., zx]T , and the output layer is a probability distribution y = [y1, y2]T over two classes.
* (ii) Plot the average training cross-entropy loss as shown below on the y-axis versus the number of hidden units on the x-axis. Explain the effect of numbers of hidden units.

## Task 3: Classification Quality Evaluation

#### (a) Explore how the size of the training data set affects the test and train accuracy. 
For each value of m in [0.1, 0.3, 0.5, 0.7, 0.9], train the classifier on the first m portion of the training examples (that is, use the data given by XTrain[0:mN] and yTrain[0:mN]). 
* (i) Report the plot of training accuracy for each such value of m with the x-axis referring to m and the y-axis referring to the classification accuracy in 𝐹1 measure as shown below;
* (ii) Report the plot of testing accuracy. 
In total, there are four curves for training accuracy and four curves for testing accuracy. Explain the general trend of the two plots in terms of training and testing accuracy if any.
𝐹1 = 2 · (𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 × 𝑅𝑒𝑐𝑎𝑙𝑙) / (𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 + 𝑅𝑒𝑐𝑎𝑙𝑙)

#### (b) Use 5-fold cross-validation to assess model performance. 
* Investigate the impact of key hyperparameters of my choices for each classifier using a testing dataset. Take SVM as an example, the classification accuracy may be significantly affected by the kernels and hyperparameter combination. List hyperparameters for each classifier and demonstrate how these hyperparameters impact on the testing accuracy. 

#### (c) Report and compare your LR, NB, SVM, and NN classifiers with the best hyperparameter settings. 
* Summarize what I have observed in the classification accuracy in 𝐹1 measure on the testing dataset. 
