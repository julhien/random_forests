# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 17:44:13 2017

@author: Mathilde
"""
from sklearn import tree
import csv
import numpy as np
import pandas

f = open( "sonar.txt", "r" )
X = []
Y = []


with open('sonar.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        X.append(row)

#for r in range(len(X)):
#    for i in range(len(X[r])-1):
#        X[r][i]=float(X[r][i])
#    Y.append(X[r][len(X[r])-1])
#    del X[r][-1]

df = pandas.DataFrame(X)
# Randomly sample 70% of your dataframe
res = []
for iter in range(1):
    df_train = df.sample(frac=0.9)
    df_test = df.loc[~df.index.isin(df_train.index)]
    Y_train = df_train[:][60]
    X_train = df_train.drop(df_train.columns[60], axis=1)
    Y_test = df_test[:][60]
    X_test = df_test.drop(df_test.columns[60], axis=1)
    
    F = 1
    t=[]
    for arbre in range(100):
        clf = tree.DecisionTreeClassifier(max_features = F)
        clf = clf.fit(X_train, Y_train)
        t.append(clf.score(X_test,Y_test))
    res.append(sum(t)/len(t))
    
r = sum(res)