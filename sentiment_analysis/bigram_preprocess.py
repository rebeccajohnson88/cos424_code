
# coding: utf-8

# In[2]:

import sklearn
from sklearn.metrics import confusion_matrix


# In[3]:

import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy
import re
import sys
import getopt
import codecs
import time
import os
from collections import Counter
import csv

chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']




# In[4]:

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


# In[6]:

## set path and import the training data
## and see the output of the tokenize corpus step
os.chdir(os.path.expanduser('~/Dropbox/ml_assignment1_share/'))
train_df = open("train.txt", 'r')
traindf_lines = train_df.readlines()

## modifying the code in the tokenize_corpus
## function from the preprocess script
## to create bigrams instead of 1-grams
def create_bigram_corpus(textdf_lines):
    
    ## read lines of textdf (already opened)
    
    
    ## store different text processing functions
    porter = nltk.PorterStemmer() # also lancaster stemmer
    wnl = nltk.WordNetLemmatizer()
    stopWords = stopwords.words("english")
    
    ## create storage objects
    docs_tokenform = [] # used for both training and test
    bigrams = [] # used for both training and test
                        
    
    ## iterate through lines of 
    ## training or test set
    for i in range(len(textdf_lines)):
        
        one_line = textdf_lines[i]
        
        # lines of the text (excludes first column
        # that contains samples and last that contains label)
        raw = ' '.join(one_line.rsplit()[1:-1])
        
        # code from pre-process file remove noisy characters; tokenize
        raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
        tokens = word_tokenize(raw)
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if w not in stopWords]
        tokens = [wnl.lemmatize(t) for t in tokens]
        tokens = [porter.stem(t) for t in tokens] 

        ## iterate through tokens to create bigrams
        ## by joining t-th word with t+1
        for t in range(0, len(tokens)-1):
            next_token = t+1
            one_bigram = ('_'.join([tokens[t], tokens[next_token]]))
            bigrams.append(one_bigram)
                    
        docs_tokenform.append(tokens)
        
    return(sorted(set(bigrams)), docs_tokenform)
  
        
## apply function to training df to get set of all unique bigrams
(bigrams, docs_tokenform) = create_bigram_corpus(traindf_lines)
            


# In[7]:

## function to create bigram representation of each document
## modified following function in preprocess code:
def create_docs_bigramform(docs_tokenform):
    bigram_docs = []
    for i in range(0, len(docs_tokenform)):
        one_doc = docs_tokenform[i]
        all_bigrams = []
        for t in range(0, len(one_doc)-1):
            next_token = t+1
            bigram_create = ('_'.join([one_doc[t], one_doc[next_token]]))
            all_bigrams.append(bigram_create)
        bigram_docs.append(all_bigrams)
    return(bigram_docs)
    


# In[8]:

## apply function to create bigram docs
docs_bigramform = create_docs_bigramform(docs_tokenform)



# In[9]:

## create function to make a dictionary
## of bigrams where key is a bigram
## and value is the count
## modified following function in preprocess code:
def create_bigram_dict(docs_bigramform):
    bigram_dictionary = dict()
    for i in range(0, len(docs_bigramform)):
        one_doc = docs_bigramform[i]
        for t in one_doc:
    
            try:
                bigram_dictionary[t] = bigram_dictionary[t] +1
            except:
                bigram_dictionary[t] = 1
    return(bigram_dictionary)

bigram_dictionary = create_bigram_dict(docs_bigramform)

    


# In[10]:

def keep_bigrams_function(bigram_dictionary):
    keep_bigrams = []
    for bigramkey in bigram_dictionary.keys():
        if(bigram_dictionary[bigramkey] > 1):
            keep_bigrams.append(bigramkey)
    print("vocab length:", len(keep_bigrams))
    return(sorted(set(keep_bigrams)))
    

bigram_vocab = keep_bigrams_function(bigram_dictionary)

## write vocab file
## write bigram vocab file
bigram_vocab_file = "bigram_vocab_update0226.txt"
outfile = codecs.open(bigram_vocab_file, 'w',"utf-8-sig")
outfile.write("\n".join(bigram_vocab))
outfile.close()


# In[11]:

## function to create bigram document term
## matrix
def create_bigram_dtm(docs_bigramform, bigram_vocab):
    bigram_dtm = numpy.zeros(shape=(len(docs_bigramform),len(bigram_vocab)), dtype=numpy.uint8)
    bigram_index = {}
    for v in range(len(bigram_vocab)):
        bigram_index[bigram_vocab[v]] = v

    for i in range(len(docs_bigramform)):
        doc = docs_bigramform[i]
        for t in doc:
            index_bigram = bigram_index.get(t)
            if index_bigram is not None: #might be python3 specific
                bigram_dtm[i,index_bigram]=bigram_dtm[i,index_bigram] +1 
    return(bigram_dtm)

bigram_dtm = create_bigram_dtm(docs_bigramform, bigram_vocab) # to double check, column sums (should be more than >= 3)

## write to file
with open("bigram_train_dtm_lowerthres.csv", "w") as f: #for python 3, modified wb to w
    writer = csv.writer(f)
    writer.writerows(bigram_dtm)
                
    


# In[12]:

## repeat process for test data

## load test df and turn into line form
testdf = open("test.txt", 'r')
testdf_lines = testdf.readlines()

## create token form of the test docs, but not the bigram
## vocabulary- function returns both but will only use first
(bigrams_test, docs_tokenform_test) = create_bigram_corpus(testdf_lines)

## then create bigram form of the test docs
docs_bigramform_test = create_docs_bigramform(docs_tokenform_test)

## create document-term matrix for
## the test docs based on 
## bigram vocabulary from the training docs
bigram_dtm_test = create_bigram_dtm(docs_bigramform_test, bigram_vocab)



# In[13]:

## double checking structure by printing excerpts
print(docs_tokenform_test[0:5])
print(docs_bigramform_test[0:5])
print(bigram_dtm_test.shape)


# In[14]:

## write test dtm to file
with open("bigram_test_dtm_lowerthres.csv", "w") as f: #for python 3, modified wb to w
    writer = csv.writer(f)
    writer.writerows(bigram_dtm_test)


# In[ ]:



