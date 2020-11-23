# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 21:22:57 2020

@author: huanz
"""

import pandas as pd
import numpy as np

#%%
#load data
data_raw = pd.read_json('data/train.jsonl', orient = 'columns', lines = True)
data_raw.head()

#%%
#data preprocessing, use response only
X = data_raw['response']
y = data_raw['label']

y.replace('SARCASM', 1, inplace=True)
y.replace('NOT_SARCASM', 0, inplace = True)

#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
    sarcasm_fit = pipeline.fit(X_train, y_train)
    y_pred = sarcasm_fit.predict(X_test)
    test_accuracy = f1_score(y_test, y_pred, average = "weighted")
    train_accuracy = f1_score(y_train, sarcasm_fit.predict(X_train), average = "weighted")
    print("test accuracy score: {0:.2f}%".format(test_accuracy*100))
    print("train accuracy score: {0:.2f}%\n".format(train_accuracy*100))
    report = classification_report(y_test, y_pred)#, target_names=['NOT_SARCASM','SARCASM'])
    return test_accuracy, train_accuracy, report

def pipeline_builder(vectorizer, classifier, stop_words='english', ngram_range=(1, 3), ccp_alpha = 0.0):
    
    vectorizer.set_params(lowercase = True, 
                          stop_words=stop_words, 
                          ngram_range=ngram_range)
    
    classifier.set_params(ccp_alpha = ccp_alpha)
    
    checker_pipeline = Pipeline([('vectorizer', vectorizer),
                                 ('classifier', classifier)])
    
    return accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)

#%%
tfidf = TfidfVectorizer()

rf = RandomForestClassifier(n_estimators=300, 
                            criterion = 'entropy', 
                            class_weight = 'balanced', 
                            random_state = 0)

#%%
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