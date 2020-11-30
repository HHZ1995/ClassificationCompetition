# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 21:22:57 2020

@author: huanz
"""

import pandas as pd
import numpy as np
import os

#%%
#load data
data_raw = pd.read_json('data/train.jsonl', orient = 'columns', lines = True)
test_raw = pd.read_json('data/test.jsonl', orient = 'columns', lines = True)
test_X = test_raw['response']
test_ID = test_raw[['id']]
#%%
#data preprocessing, use response only
X = data_raw['response']
y = data_raw['label']

y.replace('SARCASM', 1, inplace=True)
y.replace('NOT_SARCASM', 0, inplace = True)

#split data
from sklearn.model_selection import train_test_split
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size = 0.25, random_state = 0)

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def accuracy_summary(pipeline, X_train, y_train, X_eval, y_eval):
    sarcasm_fit = pipeline.fit(X_train, y_train)
    y_pred = sarcasm_fit.predict(X_eval)
    test_accuracy = f1_score(y_eval, y_pred, average = "weighted")
    train_accuracy = f1_score(y_train, sarcasm_fit.predict(X_train), average = "weighted")
    print("test accuracy score: {0:.2f}%".format(test_accuracy*100))
    print("train accuracy score: {0:.2f}%\n".format(train_accuracy*100))
    report = classification_report(y_eval, y_pred)#, target_names=['NOT_SARCASM','SARCASM'])
    return test_accuracy, train_accuracy, report

def generate_output(pipeline, X, y, test_X, test_ID, path):
    sarcasm_fit = pipeline.fit(X, y)
    y_pred = pd.Series(sarcasm_fit.predict(test_X))
    y_pred.replace(1, "SARCASM", inplace=True)
    y_pred.replace(0, 'NOT_SARCASM', inplace = True)
    test_ID['prediction'] = y_pred
    test_ID.to_csv(path_or_buf = path + '\\answer.txt', index = False, header = False)


def pipeline_builder(vectorizer, classifier, stop_words='english', ngram_range=(1, 1)):
    
    vectorizer.set_params(lowercase = False, 
                          stop_words=stop_words, 
                          ngram_range=ngram_range)
    
    checker_pipeline = Pipeline([('vectorizer', vectorizer),
                                 ('classifier', classifier)])
    
    return checker_pipeline

#%%
#define vectorizer
tfidf = TfidfVectorizer()

#define classifier
rf = RandomForestClassifier(n_estimators=300, 
                            criterion = 'entropy', 
                            class_weight = 'balanced',
                            ccp_alpha = 0,
                            random_state = 0)

svm = SVC(C = 1.0,
          kernel = "rbf",
          class_weight = 'balanced')

LR = LogisticRegression(penalty='l2',
                        class_weight='balanced')
          


#%% 
#parameter tuning
'''
prun_range = np.arange(0, 0.005, 0.0005)
trains = []
tests = []

for p in prun_range:
    print("ccp_alpha: {0:.4f}".format(p))
    test_accuracy, train_accuracy, report = pipeline_builder(vectorizer = tfidf, classifier = rf, ccp_alpha = p)
    trains.append(train_accuracy)
    tests.append(test_accuracy)

import matplotlib.pyplot as plt
plt.plot(prun_range, trains, prun_range, tests)
'''
#%%
path = os.getcwd()
pipeline = pipeline_builder(vectorizer = tfidf, classifier = LR)
generate_output(pipeline, X, y, test_X, test_ID, path)
test_accuracy, train_accuracy, report = accuracy_summary(pipeline, X_train, y_train, X_eval, y_eval)