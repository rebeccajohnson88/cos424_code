# cos424_code

Code for Princeton COS424: Fundamentals of Machine Learning (not public because instructor might repeat assignments across years)

- *Sentiment classification assignment*: code is joint with https://github.com/dmstanescu.

In this assignment, we were provided with a training set of reviews from Amazon, yelp, and other sources. This training set had two sentiment labels: positive or negative. After performing text pre-processing on the reviews (e.g., stemming; removing stopwords), and representing the reviews as a document-term matrix where each row is one review and each column is one unigram term (which relies on a bag-of-words assumption that does not encode the order of words into the feature representation), two scripts perform the classification:

1. bigram_preprocess.py: this file should be called in Python 3. In the code, we 	relax the assumption of presenting the terms in each document as unigrams. The 		code creates a universe of bigrams present in the union of all reviews, filters 	these bigrams to ones appear at least twice in 	the corpus of reviews, and then 	counts the occurrence of each bigram in each review. The output is a document-term 	matrix where each term/column, instead of being a unigram, is a bigram.

2. main_models.py: this file should be called in Python 2.7. It:
	1. Takes in the output files generated as part of pre-processing 			(document-term 	representation of training and test set, sample labels, 		etc.)
	2. Estimates the main analyses presented in the report (feature selection; 		fitting classification models; evaluating the classification models)— to 		make this step more efficient, we create a function that takes in 1. A 			list of classification models, 2. Training data/labels, 3. Test data, and 		then iterates through the classification models and stores the 
	predicted labels for the test data. This helps us flexibly switch between 		training and test datasets (for instance, ones on which we have performed 		feature selection prior to model fitting) and also flexibly add new 			classification models.
	3. Outputs the following files:
		- modelaccuracy_*: a .csv format file with model accuracy metrics 			for the three types of feature selected data (manual, lasso, 				logit)
		- modeltime_all: a .csv format file with model runtime metrics for 			the three types of feature selected models
	
3. *confusion_summary.pdf*: this summarizes four confusion matrix metrices across the classification models: true negatives (reviews with an actual negative label predicted to have a negative label); false positives (reviews with an actual negative label predicted to have a positive label); false negatives (reviews with an actual positive label predicted to have a negative label); true positives (reviews with an actual positive predicted to have positive). 

4.  *classification_results_writeup.pdf*: This write-up summarizes the main findings of the classification task, and delves deeper into one classification method: logistic regression with an L1 penalty (a form of regularization that introduces sparsity in the estimated parameters). It summarizes the goals in the task, why we care more about avoiding false positives than avoiding false negatives, and the main results.


- *Fragile families challenge assignment*: 

In this assignment, we used data from the Fragile Families cohort for children ages 0-9 to predict outcomes at age 15. 

1. ffchallenge_code.py: this code does the following:

	1. Takes in the covariate data and labels 
	2. Imputes missing data to the mean of the column
	3. Encodes categorical variables so that they can be used in an ML model
	4. Separates data into training and test set
	5. Creates three types of features: mother-reported features, father-reported features, PCA-generated features
	6. Creates a function "model_fit_function" that predicts based on the training features, calculates the MSE (the accuracy metric of interest) based on the test features, and returns the model coefficients when applicable
	7. Interfaces with the machine-readable fragile families codebook to get the question text for most important features 


