
# coding: utf-8

# In[ ]:


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
from sklearn import linear_model
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import ElasticNetCV
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import random
random.seed(4182017)
import fancyimpute
from fancyimpute import KNN
from sklearn.feature_selection import VarianceThreshold





## directories and loading files
# set working directory, print files
os.chdir(os.path.expanduser('~/Dropbox/mlassignment2/FFchallenge_v4/'))

# load covariates data
all_covariates = pd.read_csv("background.csv", delimiter = ",", low_memory = False)


# load txt file that has 
# variables that don't vary 
constant_vars = loadtxt("constantVariables.txt", dtype = np.str)


# load training df
train_df_full = pd.read_csv("train.csv", delimiter = ",")



# In[ ]:


## print the first few rows
## and dimensions
all_covariates[:5]
constant_vars[0:5]
print(all_covariates.shape)
print(train_df_full.shape)


# In[ ]:


## subset to exclude the columns that do not vary
## across individuals and 
## name new df: all_covar_vary
## leaves us with 10,000 columns

all_covar_vary = all_covariates.drop(columns = constant_vars)
print(all_covar_vary.shape)



# In[ ]:


## replace missing values

## instructor's script for imputing missing
## modified so that it skips the read.csv step
## and instead of writing the file, 
## returns the complete df
def fillMissing(df):
    
    # read input csv - takes time
    #df = pd.read_csv(inputcsv, low_memory=False)
    # Fix date bug
    df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
    
    # replace NA's with mode
    df = df.fillna(df.mode().iloc[0])
    # if still NA, replace with 1
    df = df.fillna(value=1)
    # replace negative values with 1
    num = df._get_numeric_data()
    num[num < 0] = 1
    # write filled outputcsv
    return(df)
    

## apply function
all_covar_filled = fillMissing(all_covar_vary)


# In[ ]:


### extension (predicting the predictions bullet point): split df into internal
### training set before fitting the models below

#### sample 80% as internal training df
train_df_outcomes = train_df_full.sample(frac = 0.8)

## remaining 20% as internal test df
test_df_outcomes = train_df_full.loc[~train_df_full.index.isin(train_df_outcomes.index)]

## subset features into internal test and training based on challenge ID
train_df_features = all_covar_filled.loc[all_covar_filled['challengeID'].isin(train_df_outcomes['challengeID'])]
test_df_features = all_covar_filled.loc[all_covar_filled['challengeID'].isin(test_df_outcomes['challengeID'])]




# In[ ]:


## order the outcomes and labels by challenge ID
train_features_sorted = train_df_features.sort_values(by='challengeID')
train_outcomes_sorted = train_df_outcomes.sort_values(by = 'challengeID')
test_features_sorted = test_df_features.sort_values(by='challengeID')
test_outcomes_sorted = test_df_outcomes.sort_values(by = 'challengeID')

## pre-process categorical values
## using label encoding
## make sure not to change challenge ID
encode_categorical = preprocessing.LabelEncoder()
train_features_sorted_num_noid = train_features_sorted[train_features_sorted.columns.difference(['challengeID'])].apply(encode_categorical.fit_transform)
train_features_sorted_num = pd.concat([train_features_sorted['challengeID'],
                                     train_features_sorted_num_noid], axis = 1)

## apply same encoding to internal test features
test_features_sorted_num_noid = test_features_sorted[test_features_sorted.columns.difference(['challengeID'])].apply(encode_categorical.fit_transform)
test_features_sorted_num = pd.concat([test_features_sorted['challengeID'],
                                     test_features_sorted_num_noid], axis = 1)




## remove columns in training and test with zero variance
variance_thres = VarianceThreshold()
train_features_variancethres = variance_thres.fit(train_features_sorted_num)
train_features_sorted_num_somevar_noNorm = train_features_sorted_num.loc[:, train_features_variancethres.get_support()]
test_features_sorted_num_somevar_noNorm = test_features_sorted_num.loc[:, train_features_variancethres.get_support()]

## remove ID column before normalize
train_features_noID_fornorm = train_features_sorted_num_somevar_noNorm.drop('challengeID',
                                                                           axis = 1)
test_features_noID_fornorm = test_features_sorted_num_somevar_noNorm.drop('challengeID',
                                                                           axis = 1)

train_features_prepfornorm = train_features_noID_fornorm.values
test_features_prepfornorm = test_features_noID_fornorm.values

min_max_scaler = preprocessing.MinMaxScaler()
train_features_scaled = min_max_scaler.fit_transform(train_features_prepfornorm)
train_features_sorted_num_somevar = pd.concat([train_features_sorted_num['challengeID'].reset_index(drop = True),
                                        pd.DataFrame(train_features_scaled, 
                                                 columns = train_features_noID_fornorm.columns)],
                                              axis = 1)
test_features_scaled = min_max_scaler.fit_transform(test_features_prepfornorm)
test_features_sorted_num_somevar = pd.concat([test_features_sorted_num['challengeID'].reset_index(drop = True),
                                        pd.DataFrame(test_features_scaled, 
                                                 columns = test_features_noID_fornorm.columns)],
                                              axis = 1)




# In[ ]:


## Before fitting models, perform feature selection
import re 
### five types of data
### 1. full features (train|test_features_sorted)

### 2. mother-only (grep on variables containing prefix "m")
mom_report_cols = [c for c in train_features_sorted_num_somevar if c.startswith('m')]
train_features_mom_all = pd.concat([train_features_sorted_num_somevar['challengeID'],
                                    train_features_sorted_num_somevar[mom_report_cols]], axis = 1)
test_features_mom_all = pd.concat([test_features_sorted_num_somevar['challengeID'],
                                   test_features_sorted_num_somevar[mom_report_cols]], axis = 1)


### 3. father-only (grep on variables containing prefix "f")
dad_report_cols = [c for c in train_features_sorted_num_somevar if c.startswith('f')]
train_features_dad_all = pd.concat([train_features_sorted_num_somevar['challengeID'],
                                    train_features_sorted_num_somevar[dad_report_cols]], axis = 1)
test_features_dad_all = pd.concat([test_features_sorted_num_somevar['challengeID'],
                                   test_features_sorted_num_somevar[dad_report_cols]], axis = 1)


### 4. PCA (less interpretable)- iterate through
### different N components until maximum variance explained
### code adapted from Greg Gunderson pca.py script on github
enough_pcs = []
for pcs in range(80, 100):
    pca_param = PCA(n_components = pcs)
    train_features_pca = pca_param.fit(train_features_sorted_num_somevar)
    most_variance = pca_param.explained_variance_ratio_.cumsum()[-1]
    if most_variance > 0.9:
        print('Exceeded variance with %s components' % pcs)
        enough_pcs.append(pcs)
        break

### perform pca with that number of sufficient components
PCA_final = PCA(n_components = enough_pcs[0])
train_features_pca_fit = PCA_final.fit(train_features_sorted_num_somevar)
train_features_pca = pd.concat([train_features_sorted_num_somevar['challengeID'].reset_index(drop = True),
                        pd.DataFrame(data = PCA_final.transform(train_features_sorted_num_somevar)).reset_index(drop = True)],
                               axis = 1)
test_features_pca_fit = PCA_final.fit(test_features_sorted_num_somevar)
test_features_pca = pd.concat([test_features_sorted_num_somevar['challengeID'].reset_index(drop = True),
                        pd.DataFrame(data = PCA_final.transform(test_features_sorted_num_somevar)).reset_index(drop = True)],
                               axis = 1)


## 5. mother in recent wave
mom_report_cols_recent = [c for c in train_features_sorted_num_somevar if c.startswith('m5')]
train_features_momrecent_all = pd.concat([train_features_sorted_num_somevar['challengeID'],
                                    train_features_sorted_num_somevar[mom_report_cols_recent]], axis = 1)
test_features_momrecent_all = pd.concat([test_features_sorted_num_somevar['challengeID'],
                                   test_features_sorted_num_somevar[mom_report_cols_recent]], axis = 1)


# In[ ]:


## save columns under each form of feature selection
feature_N_postselect = pd.DataFrame([train_features_sorted_num_somevar.shape[1],
                       train_features_mom_all.shape[1],
                       train_features_dad_all.shape[1],
                       train_features_pca.shape[1],
                    train_features_momrecent_all.shape[1]])


## save and print
feature_N_postselect.index = ["all", "mom", "father", "pca", "mom recent"]


# In[ ]:


## function to split data based on outcome 
## and subset internal training and test outcome
## data to those 

def df_nonmissing_outcomes(train_outcomes, test_outcomes, 
						   outcome_string):
	
	# subset to non-missing internal training set outcomes
	train_outcome_all = train_outcomes[['challengeID', outcome_string]]
	train_outcome_nomiss = train_outcome_all.dropna(subset = [outcome_string])
	
	# subset to non-missing internal test set outcomes
	test_outcome_all = test_outcomes[['challengeID', outcome_string]]
	test_outcome_nomiss = test_outcome_all.dropna(subset = [outcome_string])
   
	## non-missing outcomes
	return(train_outcome_nomiss, test_outcome_nomiss)


## function to then subset a given feature set
## to those not missing data for those outcomes- iterates through
## list of feature sets
def df_features_ofcompleteDV(train_complete_id,
							 test_complete_id,
						   train_feature_list,
						test_feature_list):
	
	# pull out a specific feature set
	train_feature_complete = []
	test_feature_complete = []
	for i in range(0, len(train_feature_list)):
		one_feature_set = train_feature_list[i]
		train_features_nomiss = one_feature_set.loc[one_feature_set['challengeID'].isin(train_complete_id)]
		train_feature_complete.append(train_features_nomiss)
		
	for i in range(0, len(test_feature_list)):
		one_feature_set = test_feature_list[i]
		test_features_nomiss = one_feature_set.loc[one_feature_set['challengeID'].isin(test_complete_id)]
		test_feature_complete.append(test_features_nomiss)
	
	return(train_feature_complete, test_feature_complete)


 ## list of feature sets
train_feature_list =   [train_features_sorted_num, train_features_mom_all, train_features_dad_all, train_features_pca, train_features_momrecent_all]
test_feature_list =   [test_features_sorted_num, test_features_mom_all, test_features_dad_all, test_features_pca, test_features_momrecent_all]                        

## create complete outcome data
## and three sets of feature data for gpa
(train_gpa_outcome, test_gpa_outcome) = df_nonmissing_outcomes(train_outcomes_sorted,
																 test_outcomes_sorted,
																outcome_string = 'gpa')

(train_feature_gpa, test_feature_gpa) = df_features_ofcompleteDV(train_complete_id = 
																train_gpa_outcome['challengeID'],
											test_complete_id = test_gpa_outcome['challengeID'],
											train_feature_list = train_feature_list,
											test_feature_list = test_feature_list)


## Do the same for grit 
(train_grit_outcome, test_grit_outcome) = df_nonmissing_outcomes(train_outcomes_sorted,
																 test_outcomes_sorted,
																outcome_string = 'grit')

(train_feature_grit, test_feature_grit) = df_features_ofcompleteDV(train_complete_id = 
																train_grit_outcome['challengeID'],
											test_complete_id = test_grit_outcome['challengeID'],
											train_feature_list = train_feature_list,
											test_feature_list = test_feature_list)


#Do the same for material hardship
	
(train_materialHardship_outcome, test_materialHardship_outcome) = df_nonmissing_outcomes(train_outcomes_sorted,
																 test_outcomes_sorted,
																outcome_string = 'materialHardship')

(train_feature_materialHardship, test_feature_materialHardship) = df_features_ofcompleteDV(train_complete_id = 
																train_materialHardship_outcome['challengeID'],
											test_complete_id = test_materialHardship_outcome['challengeID'],
											train_feature_list = train_feature_list,
											test_feature_list = test_feature_list)


    


# In[ ]:


def model_fit_function(model_list, training_features, 
					   training_outcome, test_features, test_outcome):
	store_mse = []
	store_yhat = []
	store_coef = []
	for i in range(0, len(model_list)):

		## pull out model
		one_model = model_list[i]

		## fit model
		fitted_model = one_model.fit(training_features, training_outcome)
		
		## print
		print 'finished fitting: %s' % one_model
		
		## predict on test features
		one_model_predicted = one_model.predict(test_features)
		
		## append predicted values
		store_yhat.append(one_model_predicted)
		
		## calculate MSE and append
		store_mse.append(mean_squared_error(test_outcome, one_model_predicted))
		
		## store coefs from the model
		if i < 3:
			store_coef.append(fitted_model.coef_)
		else:
			store_coef.append(fitted_model.feature_importances_)

	
	## return mse and predicted values
	return(store_mse, store_yhat, store_coef)
	
## try with couple alphas
## FOR GRACIE: these should use linear_model.LassoCV instead of
## specifying an alpha but mine was taking forever to run.
## Could you try to see if it runs on yours?
model_list_full = [linear_model.LassoCV(),
				   linear_model.RidgeCV(),
				   ElasticNetCV(), 
				   RandomForestRegressor(n_estimators = 1000, random_state = 42)]



## function to store models
###  save model names
model_names = ['lasso', 'ridge', 'elasticnet', 'randomforest']
feature_names = ['mother-report', 'mother-report_recent',
				'father-report', 'pca']

## function to label and write to csv
def store_results(mse_df, outcome_string):
	mse_df.columns = model_names
	mse_df.index = feature_names
	filename = 'mseresults_%s' % outcome_string
	with open(filename, "w") as f: 
		mse_df.to_csv(f, header = True)
		



### results for gps
(results_GPA_momfeatures_mse, 
 results_GPA_momfeatures_yhat,
results_GPA_momfeatures_coef) = model_fit_function(model_list = model_list_full,
							training_features = train_feature_gpa[1],
							training_outcome = train_gpa_outcome['gpa'],
							test_features = test_feature_gpa[1],
							test_outcome = test_gpa_outcome['gpa'])

(results_GPA_dadfeatures_mse, 
 results_GPA_dadfeatures_yhat,
results_GPA_dadfeatures_coef) = model_fit_function(model_list = model_list_full,
							training_features = train_feature_gpa[2],
							training_outcome = train_gpa_outcome['gpa'],
							test_features = test_feature_gpa[2],
							test_outcome = test_gpa_outcome['gpa'])

 
(results_GPA_pcafeatures_mse, 
 results_GPA_pcafeatures_yhat,
results_GPA_pcafeatures_coef) = model_fit_function(model_list = model_list_full,
							training_features = train_feature_gpa[3],
							training_outcome = train_gpa_outcome['gpa'],
							test_features = test_feature_gpa[3],
							test_outcome = test_gpa_outcome['gpa'])

(results_GPA_momrecentfeatures_mse, 
 results_GPA_momrecentfeatures_yhat,
results_GPA_momrecentfeatures_coef) = model_fit_function(model_list = model_list_full,
							training_features = train_feature_gpa[4],
							training_outcome = train_gpa_outcome['gpa'],
							test_features = test_feature_gpa[4],
							test_outcome = test_gpa_outcome['gpa'])


### save GPA results
gpa_mse = pd.concat([pd.DataFrame(results_GPA_momfeatures_mse).T,
						pd.DataFrame(results_GPA_momrecentfeatures_mse).T,
							 pd.DataFrame(results_GPA_dadfeatures_mse).T,
							 pd.DataFrame(results_GPA_pcafeatures_mse).T],
							axis = 0)
store_results(gpa_mse, 'gpa')


### repeat for grit

(results_grit_momfeatures_mse, 
 results_grit_momfeatures_yhat,
results_grit_momfeatures_coef) = model_fit_function(model_list = model_list_full,
							training_features = train_feature_grit[1],
							training_outcome = train_grit_outcome['grit'],
							test_features = test_feature_grit[1],
							test_outcome = test_grit_outcome['grit'])

(results_grit_dadfeatures_mse, 
 results_grit_dadfeatures_yhat,
results_grit_dadfeatures_coef) = model_fit_function(model_list = model_list_full,
							training_features = train_feature_grit[2],
							training_outcome = train_grit_outcome['grit'],
							test_features = test_feature_grit[2],
							test_outcome = test_grit_outcome['grit'])

 
(results_grit_pcafeatures_mse, 
 results_grit_pcafeatures_yhat,
results_grit_pcafeatures_coef) = model_fit_function(model_list = model_list_full,
							training_features = train_feature_grit[3],
							training_outcome = train_grit_outcome['grit'],
							test_features = test_feature_grit[3],
							test_outcome = test_grit_outcome['grit'])

(results_grit_momrecentfeatures_mse, 
 results_grit_momrecentfeatures_yhat,
results_grit_momrecentfeatures_coef) = model_fit_function(model_list = model_list_full,
							training_features = train_feature_grit[4],
							training_outcome = train_grit_outcome['grit'],
							test_features = test_feature_grit[4],
							test_outcome = test_grit_outcome['grit'])


grit_mse = pd.concat([pd.DataFrame(results_grit_momfeatures_mse).T,
						pd.DataFrame(results_grit_momrecentfeatures_mse).T,
							 pd.DataFrame(results_grit_dadfeatures_mse).T,
							 pd.DataFrame(results_grit_pcafeatures_mse).T],
							axis = 0)

store_results(grit_mse, 'grit')


## repeat for material hardship
(results_material_momfeatures_mse, 
 results_material_momfeatures_yhat,
results_material_momfeatures_coef) = model_fit_function(model_list = model_list_full,
							training_features = train_feature_materialHardship[1],
							training_outcome = train_materialHardship_outcome['materialHardship'],
							test_features = test_feature_materialHardship[1],
							test_outcome = test_materialHardship_outcome['materialHardship'])

(results_material_dadfeatures_mse, 
 results_material_dadfeatures_yhat,
results_material_dadfeatures_coef) = model_fit_function(model_list = model_list_full,
							training_features = train_feature_materialHardship[2],
							training_outcome = train_materialHardship_outcome['materialHardship'],
							test_features = test_feature_materialHardship[2],
							test_outcome = test_materialHardship_outcome['materialHardship'])

(results_material_pcafeatures_mse, 
 results_material_pcafeatures_yhat,
results_material_pcafeatures_coef) = model_fit_function(model_list = model_list_full,
							training_features = train_feature_materialHardship[3],
							training_outcome = train_materialHardship_outcome['materialHardship'],
							test_features = test_feature_materialHardship[3],
							test_outcome = test_materialHardship_outcome['materialHardship'])


(results_material_momrecentfeatures_mse, 
 results_material_momrecentfeatures_yhat,
results_material_momrecentfeatures_coef) = model_fit_function(model_list = model_list_full,
							training_features = train_feature_materialHardship[4],
							training_outcome = train_materialHardship_outcome['materialHardship'],
							test_features = test_feature_materialHardship[4],
							test_outcome = test_materialHardship_outcome['materialHardship'])


hardship_mse = pd.concat([pd.DataFrame(results_material_momfeatures_mse).T,
						pd.DataFrame(results_material_momrecentfeatures_mse).T,
							 pd.DataFrame(results_material_dadfeatures_mse).T,
							 pd.DataFrame(results_material_pcafeatures_mse).T],
							axis = 0)



store_results(hardship_mse, 'hardship')

print 'done running models'


## store coefs from best-fitting models
results_GPA_momrecentfeatures_coef[3].dump("gpa_coef_rf_momrecent.dat")
results_grit_momfeatures_coef[2].dump("grit_coef_elastnet_mom.dat")
results_grit_momrecentfeatures_coef[2].dump("grit_coef_elastnet_momrecent.dat")
results_material_momfeatures_coef[2].dump("material_coef_elastnet_mom.dat")
results_material_momrecentfeatures_coef[2].dump("material_coef_elastnet_momrecent.dat")


## store yhat from best fitting GPA
results_GPA_momrecentfeatures_yhat[3].dump("gpa_yhat_rf_momrecent.dat")




# In[157]:


## way to interpret one: coefs not penalized to zero
## or most important features

## define functions
import urllib2
import json


def get_descriptions(varnames):
    store_descriptions = []
    for coef in range(0, len(varnames)):
        one_coef = varnames[coef]
        new_url = 'https://codalab.fragilefamilieschallenge.org/f/api/codebook/%s' % one_coef
        all_meta = json.load(urllib2.urlopen(new_url))
        try:
            description_meta = all_meta['description']
            store_descriptions.append(description_meta)
        except:
            description_meta = "missing"
            store_descriptions.append(description_meta)
    return(store_descriptions)


def findvars_printdescrip(coefs, colnames, outcome_string):
    coefs_mostneg = np.sort(coefs)[0:9]
    lencoefs = len(coefs)
    lencoefs_cutoff = lencoefs - 10
    coefs_mostpos = np.sort(coefs)[lencoefs_cutoff:lencoefs]
    mostneg_coefs = colnames[coefs < coefs_mostneg[8]]
    mostpos_coefs = colnames[coefs > coefs_mostpos[0]]
    samplezero_coefs = random.sample(colnames[coefs == 0], 10)
    file_suffix = '_%s.txt' % outcome_string
    neg_descrip_file = 'neg_descrip%s' % file_suffix
    pos_descrip_file = 'pos_descrip%s' % file_suffix
    zero_descrip_file = 'zero_descrip%s' % file_suffix
    ## apply to the three sets of coefficients
    neg_descriptions = get_descriptions(mostneg_coefs)
    pos_descriptions = get_descriptions(mostpos_coefs)
    zero_descriptions = get_descriptions(samplezero_coefs)
    print(neg_descriptions)
    print(pos_descriptions)
    print(zero_descriptions)
    ## write files
    descrip_neg = open(neg_descrip_file, "w")
    line = descrip_neg.writelines(neg_descriptions)
    descrip_neg.close()
    descrip_pos = open(pos_descrip_file, "w")
    line = descrip_pos.writelines(pos_descriptions)
    descrip_pos.close()
    descrip_zero = open(zero_descrip_file, "w")
    line = descrip_zero.writelines(zero_descriptions)
    descrip_zero.close()



## apply to the diff outcomes - need to adapt for random forest

## mom-reported features; grit and material hardship; elastic net
grit_coef = np.load('grit_coef_elastnet_momrecent.dat')
hardship_coef = np.load('material_coef_elastnet_momrecent.dat')


## apply function
findvars_printdescrip(coefs = grit_coef, colnames = train_feature_grit[4].columns.values,
                    outcome_string = 'grit')

findvars_printdescrip(coefs = hardship_coef, colnames = train_feature_materialHardship[4].columns.values,
                    outcome_string = 'hardship')





# In[ ]:



## adapt to find most important features for random forest
gpa_featureimport = np.load('gpa_coef_rf_momrecent.dat')
len_feature = len(gpa_featureimport)
len_index = len_feature - 10
coef_largest = np.sort(gpa_featureimport)[len_index:len_feature]
coef_largest_names = train_feature_gpa[4].columns.values[gpa_featureimport > coef_largest[0]]
large_descriptions = get_descriptions(coef_largest_names)
descrip_rf = open("large_descrip_gpa.txt", "w")
line = descrip_rf.writelines(large_descriptions)
descrip_rf.close() ## not super interpretable 


# In[74]:


## way to interpret two: clustering residuals
## to see who falls below or above expectation

## read in residuals for RF 
resid_chosen = pd.DataFrame(np.load('gpa_yhat_rf_momrecent.dat'), 
                            columns = ['predicted_gpa'])

## go back to non-pruned test
## set and merge based on ID
gpa_obs_pred = pd.concat([resid_chosen.reset_index(drop = True),
                         test_gpa_outcome.reset_index(drop = True)], axis = 1)


gpa_obs_pred['residual'] = gpa_obs_pred.gpa - gpa_obs_pred.predicted_gpa
gpa_obs_pred_full = gpa_obs_pred.dropna()

## suppress warnings about doing things
## on subsetted df
pd.options.mode.chained_assignment = None

## cluster the residuals into three groups using Gaussian mixture
random.seed(4718)
gaussmix = GaussianMixture(n_components=3).fit(np.array(gpa_obs_pred_full[['residual']]).reshape(-1, 1))
labels = gaussmix.predict(np.array(gpa_obs_pred_full[['residual']]).reshape(-1, 1))
gpa_obs_pred_full['label'] = labels

## export the data to plot in ggplot
with open("residualplot_df.csv", "w") as f: 
		gpa_obs_pred_full.to_csv(f, header = True)


# In[156]:



## create a binary label of over-performer versus not
conditions = [
    (gpa_obs_pred_full['label'] == 0) | (gpa_obs_pred_full['label'] == 2),
    (gpa_obs_pred_full['label'] == 1)]
choices = [0, 1]

gpa_obs_pred_full['binary_label'] = np.select(conditions, choices, default=0)

## merge with full mom-reported features based on
## challengeID


gpa_labels_withfeatures = gpa_obs_pred_full.merge(pd.DataFrame(data = test_feature_gpa[1]),
                                                on = ['challengeID'])

print(gpa_labels_withfeatures[1:5])

## subset to features and label
gpa_labels_featuresonly = gpa_labels_withfeatures.drop(columns = ['challengeID',
                                                                 'predicted_gpa',
                                                                 'gpa',
                                                                 'residual',
                                                                 'label',
                                                                 'binary_label'])


## find most positive coefficients
gpa_labels_labelonly = gpa_labels_withfeatures['binary_label']

## run model predicting label
lr = LogisticRegression(penalty = "l1", solver='liblinear')
fit_model = lr.fit(gpa_labels_featuresonly, gpa_labels_labelonly)
label_modelcoef = fit_model.coef_

## get ten most positive coefficients


labelcoefs_all = label_modelcoef[0]
labelcoefs_mostpos_raw = np.sort(label_modelcoef)[0]
n_coefs = len(labelcoefs_mostpos_raw)
n_coef_cutoff = n_coefs - 10
labelcoefs_mostpos_final = labelcoefs_mostpos_raw[n_coef_cutoff:n_coefs]

## look up varnames and description
labelcoefs_mostpos_names = gpa_labels_featuresonly.columns.values[labelcoefs_all > labelcoefs_mostpos_final[0]]
label_descriptions = get_descriptions(labelcoefs_mostpos_names)


## write descriptions to file
beat_theodds = open("beatodds_labels.txt", "w")
line = beat_theodds.writelines(label_descriptions)
beat_theodds.close()



