
# COS424 HM1
# Rebecca Johnson and Diana Stanescu

# coding: utf-8

# In[72]:

## importing packages
import os
from os import listdir
import csv
import itertools
import numpy as np
from numpy import loadtxt
import pandas as pd
from numpy import genfromtxt
import scipy
import sklearn
import nltk
import matplotlib.pyplot as plt
print sklearn.__version__
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier




## directories and loading files
# set working directory, print files
os.chdir(os.path.expanduser('~/Dropbox/ml_assignment1_share/'))

# load bag of words data and use
# numpy to turn into an array
# then pandas to turn into a dataframe
bagwords_csv = csv.reader(open("out_bag_of_words_5.csv", "rb"), 
                          delimiter = ",")

bow_label = loadtxt("out_vocab_toread.txt", dtype = np.str)


features_train = genfromtxt('out_bag_of_words_5.csv', delimiter=',')
label_train = loadtxt("out_classes_5.txt", 
                          comments="#", delimiter="\n", unpack=False, dtype = np.uint8)
features_test = genfromtxt('out_test_bag_of_words_0.csv', delimiter=',')
label_test = loadtxt("out_test_classes_0.txt", 
                          comments="#", delimiter="\n", 
                          unpack=False, dtype = np.uint8)
features_train_df = pd.DataFrame(features_train, columns = bow_label)
features_test_df = pd.DataFrame(features_test, columns = bow_label)


# In[62]:

# Step one: feature selection

## Step 1A: manual feature selection

### look at frequencies of features in the combined data
a = np.count_nonzero(features_train, axis = 0)
plt.hist(a, bins = 'auto')
plt.show()

### concat bag of words and sentiment label

labeled_bagwords = np.column_stack((label_train, features_train))

### separate into two arrays, by classification, removing sentiment label col

positive_bagwords = labeled_bagwords[labeled_bagwords[:, 0] == 1, 1:]
negative_bagwords = labeled_bagwords[labeled_bagwords[:, 0] == 0, 1:]

### calculate frequencies

#### positive
p = np.count_nonzero(positive_bagwords, axis = 0)

#### negative
n = np.count_nonzero(negative_bagwords, axis = 0)

### since there is a 50/50 split on number of observations between n and p
### no need to normalize by number of observations in each array

diff = abs(p - n)
print(diff.shape)

#### remove features with diff < 3

features_selected = features_train[:, diff[:] > 2] # removed 185 features
features_test_selected = features_test[:, diff[:] > 2] 

print(features_selected.shape)

#### this is very similar to the Categorical Proportional Difference (PD) 

pd = abs(p - n)/(p+n)

# however, using pd would reduce our features to 46 (!), even for low threshold
# so we decided to apply a less stringent feature selection, but will use this data space 
# for validation too

bagwords_pd = features_train[:, pd[:] > 0.2]
features_test_pd = features_test[:, pd[:] > 0.2]

print(bagwords_pd.shape)

### print bagwords selected
### maybe rename to manual feature selection


## Step 1B: model-based feature selection


### lasso
lasso = SelectFromModel(LassoCV()) # default threshold is mean
lasso.fit(features_selected, label_train)

features_l1 = lasso.transform(features_selected) # 273 features
features_test_l1 = lasso.transform(features_test_selected) 

print(features_l1.shape)

### logit 

logit = SelectFromModel(LogisticRegressionCV(solver = "liblinear", 
                                 penalty = "l1"))
logit.fit(features_selected, label_train)

features_logit = logit.transform(features_selected) 
features_test_logit = logit.transform(features_test_selected)


# In[94]:

# Step two: fit classifier models on three feature spaces

## define function that takes in classification models and feature spaces
## and returns the predicted y for each
def classification_models(model_list, training_features, test_features):

	store_yhat = []
	for i in range(0, len(model_list)):

		## pull out model
		one_model = model_list[i]

		## fit model
		one_model.fit(training_features, label_train)

		## find yhat and append
		store_yhat.append(one_model.predict(test_features))

	return(store_yhat)


## create a list of model objects
classifiers_list = [DecisionTreeClassifier(random_state=0), RandomForestClassifier(random_state = 0),
				AdaBoostClassifier(), RidgeClassifier(), SGDClassifier(random_state = 22),
				SGDClassifier(loss = "squared_hinge", random_state = 22), 
                SGDClassifier(random_state = 22, penalty = "l1"),
				SGDClassifier(loss = "squared_hinge", random_state = 22,
                      penalty = "l1"), 
				SGDClassifier(random_state = 22, penalty = "elasticnet"),
				SGDClassifier(loss = "squared_hinge", random_state = 22,
                      penalty = "elasticnet"),
				PassiveAggressiveClassifier(random_state = 22, 
                                 loss = "squared_hinge"),
				CalibratedClassifierCV(),
				LogisticRegression(),
				LogisticRegressionCV(),
				LogisticRegression(penalty = "l1"),
				LogisticRegressionCV(solver = "liblinear", 
                                 penalty = "l1")]

names_list = ['dt', 'rf', 'ada', 'ridge', 'svml', 'svmsqhinge', 'svml1', 'svmsqhingel1',
              'svme', 'svmesqhinge', 'pa', 'calibrated', 'logit', 'logitcv', 'logitl1',
              'logitl1cv']

model_list = classifiers_list+[VotingClassifier(estimators=zip(names_list, classifiers_list), voting = 'hard')]

## feed the function the list of models and 
## apply to the three types of feature sets

### feature set 1: manual selection
yhat_manualselect = classification_models(model_list, features_selected,
                    features_test_selected)

### feature set 2: manual selection + logit
yhat_logitselect = classification_models(model_list, features_logit,
                   features_test_logit)

### feature set 3: manual selection + lasso
yhat_lassoselect = classification_models(model_list, features_l1,
                   features_test_l1)


# In[64]:

# Step three: evaluate accuracy

## define function to evaluate model accuracy
def all_scores(y_pred):
    confusion_matrix_results = confusion_matrix(label_test, y_pred)
    classification_report_results = classification_report(label_test, y_pred)
    roc_curve_results = roc_curve(label_test, y_pred, pos_label = 1)
    roc_auc_results = roc_auc_score(label_test, y_pred)
    return(confusion_matrix_results, classification_report_results, roc_curve_results, roc_auc_results)


## apply model accuracy function to
## each of the types of predicted y
store_accuracy_manual = []
for i in range(0, len(yhat_manualselect)):

	one_yhat = yhat_manualselect[i]
	one_yhat_acc = all_scores(one_yhat)
	store_accuracy_manual.append(one_yhat_acc)

print(store_accuracy_manual)

store_accuracy_logitselect = []
for i in range(0, len(yhat_logitselect)):

	one_yhat = yhat_logitselect[i]
	one_yhat_acc = all_scores(one_yhat)
	store_accuracy_logitselect.append(one_yhat_acc)


store_accuracy_lassoselect = []
for i in range(0, len(yhat_lassoselect)):

	one_yhat = yhat_lassoselect[i]
	one_yhat_acc = all_scores(one_yhat)
	store_accuracy_lassoselect.append(one_yhat_acc)


## write the model accuracy results to a .csv file
with open("modelaccuracy_manual.csv", "w") as f: 
    writer = csv.writer(f)
    writer.writerows(store_accuracy_manual)


with open("modelaccuracy_lasso.csv", "w") as f: 
    writer = csv.writer(f)
    writer.writerows(store_accuracy_lassoselect)

with open("modelaccuracy_logit.csv", "w") as f: 
    writer = csv.writer(f)
    writer.writerows(store_accuracy_logitselect)


# In[65]:

# Step four: extension to bigrams

### Skips features selection

# loading files

features_test_bigram = genfromtxt("bigram_test_dtm.csv", 
                          delimiter = ",")
features_train_bigram = genfromtxt("bigram_train_dtm.csv",
                          delimiter = ",")

# get predictions for bigrams
yhat_bigrams = classification_models(model_list, features_train_bigram,
                    features_test_bigram)


# In[66]:

# accuracy

store_accuracy_bigrams = []
for i in range(0, len(yhat_bigrams)):

	one_yhat = yhat_bigrams[i]
	one_yhat_acc = all_scores(one_yhat)
	store_accuracy_bigrams.append(one_yhat_acc)

print(store_accuracy_bigrams)

## write the model accuracy results to a .csv file
with open("modelaccuracy_bigrams.csv", "w") as f: #for python 3, modified wb to w
    writer = csv.writer(f)
    writer.writerows(store_accuracy_bigrams)


# In[89]:

# Analysis with lower threshold bigrams

features_test_bigram = genfromtxt("bigram_test_dtm_lowerthres.csv", 
                          delimiter = ",")
features_train_bigram = genfromtxt("bigram_train_dtm_lowerthres.csv",
                          delimiter = ",")

# get predictions for bigrams
yhat_bigrams = classification_models(model_list, features_train_bigram,
                    features_test_bigram)

# accuracy

store_accuracy_bigrams = []
for i in range(0, len(yhat_bigrams)):

	one_yhat = yhat_bigrams[i]
	one_yhat_acc = all_scores(one_yhat)
	store_accuracy_bigrams.append(one_yhat_acc)

print(store_accuracy_bigrams)

## write the model accuracy results to a .csv file
with open("modelaccuracy_bigrams_lowerthres.csv", "w") as f: #for python 3, modified wb to w
    writer = csv.writer(f)
    writer.writerows(store_accuracy_bigrams)


# In[73]:

## code for the feature interpretation
## wrt spotlight classifier

# assign column names
# to understand features
# pruned at each feature selection step


## manual selection on labeled
## data.frame version of df
## repeat selection on labeled data.frame version
features_selected_df = features_train_df.iloc[:, diff[:] > 2] # removed 185 features
features_test_selected_df = features_test_df.iloc[:, diff[:] > 2] 

## get column names left out by manual selection
features_original_col = features_train_df.columns.values
features_postmanual_col = features_selected_df.columns.values

## get indicators of columns selected out by lasso 
lasso_features_indicators = lasso.fit(features_selected, label_train).get_support()

## use the feature indicators to subset 
## which colummns from the manual selection are removed from data
## used to estimate model0 list of column
## names of features fed to classificaiton models
features_postlasso_col = features_postmanual_col[lasso_features_indicators]


## fit spotlight model outside above function
## fit the spotlight model in order to 
## find exemplar documents and words
spotlight_lcv = LogisticRegressionCV(solver = "liblinear", 
                                 penalty = "l1")

spotlight_lcv_fit = spotlight_lcv.fit(features_l1, label_train)

## find yhat for spotlight model
spotlight_yhat_binary = spotlight_lcv_fit.predict(features_test_l1)

## find predicted probs for spotlight model
spotlight_yhat_nonbin = spotlight_lcv_fit.predict_proba(features_test_l1)

## binary returns 0, 1
## non-binary returns predicted values
spotlight_yhat_neglabel = spotlight_yhat_nonbin[:, 0]
spotlight_yhat_poslabel = spotlight_yhat_nonbin[:, 1]

## return index of reviews with highest prob neg
## and highest prob pos
spotlight_highprobneg_index = spotlight_yhat_neglabel.tolist().index(max(spotlight_yhat_neglabel))
spotlight_highprobpos_index = spotlight_yhat_poslabel.tolist().index(max(spotlight_yhat_poslabel))


## return review from original list
## of review lines
test_samples = open("test.txt", 'r')
test_lines = test_samples.readlines()

test_mostneg = test_lines[spotlight_highprobneg_index]
print(test_mostneg)

test_mostpos = test_lines[spotlight_highprobpos_index]
print(test_mostpos)

## find example of beta penalized towards 
## zero versus not
getcoefs_spotlight = np.transpose(spotlight_lcv_fit.coef_)[:, 0]
#length 273 vector of coefficient values

## find most positive and negative
## coef and get indices of them
mostneg_coef = np.sort(getcoefs_spotlight)[0:4]
len_coef = len(getcoefs_spotlight)
mostpos_coef = np.sort(getcoefs_spotlight)[(len_coef-4):len_coef]

## highlight coefs with value zero from this step
print(features_postlasso_col[abs(getcoefs_spotlight) < 0.1])

## highlight features with large negative coefficient
print(features_postlasso_col[getcoefs_spotlight <= mostneg_coef[len(mostneg_coef)-1]])
print(features_postlasso_col[getcoefs_spotlight >= mostpos_coef[0]])




# In[93]:

## function to estimate time
## to run model on each 
import time

## re-estimate each model and return the time
## needed to run each
def runtime_classification_models(model_list, training_features, test_features):
    
    store_time = []
    for i in range(0, len(model_list)):
        
        time_start = time.time()
        ## pull out model
        one_model = model_list[i]
        
        ## fit model
        fit_model = one_model.fit(training_features, label_train)
        
        ## find yhat and append
        run_time = time.time() - time_start
        store_time.append(run_time) 
    
    return(store_time)


## apply to model list
## and features
### feature set 1: manual selection
runtime_manualselect = runtime_classification_models(model_list, features_selected,
                    features_test_selected)

### feature set 2: manual selection + logit
runtime_logitselect = runtime_classification_models(model_list, features_logit,
                   features_test_logit)

### feature set 3: manual selection + lasso
runtime_lassoselect = runtime_classification_models(model_list, features_l1,
                   features_test_l1)

# concat lists
runtime_models = np.column_stack((runtime_manualselect, runtime_logitselect, runtime_lassoselect))


with open("modeltime_all.csv", "w") as f: #for python 3, modified wb to w
    writer = csv.writer(f)
    writer.writerows(runtime_models)


# In[ ]:



