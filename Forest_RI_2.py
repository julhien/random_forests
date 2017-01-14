# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 17:44:13 2017

@author: Mathilde
"""

import decTree_RC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import csv
import numpy as np
import pandas
import glob
import warnings
import decTree
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class Forest:
    ##sk_learn determines if we use sk_learn
    ##with L=1 default , RI used, else RC
    def __init__(self, forest_size, sk_learn, F , L=1):
        self.trees = []
        self.forest_indices = []
        self.forest_size = forest_size
        self.sk_learn = sk_learn
        self.F = F
        self.L = L
    def train(self, df_train):
        i = 0
        for arbre in range(self.forest_size):
            i+=1
            if i%10==0:
                print(i)
            # bagging
            df_train_bagged = df_train.sample(frac=1., replace=True)
            self.forest_indices.append(df_train_bagged.index)
            if self.sk_learn:
                ##Construct a forest thanks to sklearn

                Y_train = df_train_bagged.drop(df_train_bagged.columns[:len(df_train.T) - 1], axis=1)
                X_train = df_train_bagged.drop(df_train_bagged.columns[len(df_train.T) - 1], axis=1)
                clf = tree.DecisionTreeClassifier(max_features=self.F)
                self.trees.append(clf.fit(X_train, Y_train))
            # construct a forest with decision tree
            else:
                if self.L==1:
                    tree = decTree.Node()

                    tree.train(df_train_bagged.values.tolist(), 150 , 5, self.F)
                    self.trees.append(tree)
                else:

                    tree = decTree_RC.Node()
                    tree.train(df_train_bagged.values.tolist(), 150, 5, self.F, self.L)
                    self.trees.append(tree)
##Normalize set except last column - returns mean and std: meant for training set with all columns given
def normalize_train(df):
    i = 0
    mean = []
    std = []
    for column in df:
        if i<len(df.columns)-1:
            mean.append(df[column].mean())
            if df[column].std()!=0:
                std.append(df[column].std())
            else:
                std.append(1)
            df[column] = df[column].apply(lambda x : (x - mean[i]) / std[i])
        i+=1
    return df, mean, std

##Normalize all set, given mean and std
def normalize_test(X, mean, std):
    i = 0
    for column in X:
            X[column] = X[column].apply(lambda x : (x-mean[i])/std[i])
            i+=1
    return X

def convert_to_float(data):
    for r in range(len(data)):
        for i in range(len(data[r])-1):
            data[r][i]=float(data[r][i])
    return data

def function_test_set_error(X_test,Y_test,forest):
    error = 0.
    for x in range(X_test.shape[0]):
        #We use the majority vote
        vote = 0
        for arbre in range(forest.forest_size):
            if forest.sk_learn:
                vote = vote + (forest.trees[arbre].predict(X_test.iloc[[x]].as_matrix())[0]==Y_test.iloc[[x]].as_matrix())
            else:
                if forest.L ==1:
                    vote = vote + (forest.trees[arbre].predict(X_test.iloc[x].as_matrix())==Y_test.iloc[x].as_matrix())
                else:
                    vote = vote + (
                        forest.trees[arbre].predict(X_test.iloc[x].as_matrix()) == Y_test.iloc[
                        x].as_matrix())
        if (vote <int(forest.forest_size/2)):
            error = error + 1.
    return error/len(X_test)
    
def out_of_bag_error(X_t,Y_t,forest):
    error = 0.
    for x in range(X_t.shape[0]):
        #We use the majority vote
        vote = 0
        total = 0
        for arbre in range(forest.forest_size):
            #test if we consider this tree - is the data x used to construct the tree arbre
            if not X_t.index[x] in forest.forest_indices[arbre]:
                total = total + 1
                if forest.sk_learn:
                    vote = vote + (forest.trees[arbre].predict(X_t.iloc[[x]].as_matrix())[0]==Y_t.iloc[[x]].as_matrix())
                else:
                    if forest.L == 1:
                        vote = vote + (forest.trees[arbre].predict(X_t.iloc[x].as_matrix())==Y_t.iloc[x].as_matrix())
                    else:

                        vote = vote + (
                            forest.trees[arbre].predict(X_t.iloc[x].as_matrix()) == Y_t.iloc[
                            x].as_matrix())

        if (vote <int(total/2)):
            error = error + 1.

    return error/len(X_t)

def permuted_oob_error(X_t,Y_t, forest):
    errors = [0]*X_t.shape[1]
    for feature in range(X_t.shape[1]):
        vote = [0] * (max(X_t.index)+1)
        total = [0] * (max(X_t.index)+1)
        for arbre in range(forest.forest_size):
                X_t_arbre = pandas.DataFrame.copy(X_t)
                indices_oob_arbre = [i for i in X_t.index if i not in forest.forest_indices[arbre]]
                # X_t_arbre = [X_t[i] for i in range(X_t.shape[0]) if i not in forest_indexes[arbre]]
                permuted_index = np.random.permutation(indices_oob_arbre)
                for i in range(len(indices_oob_arbre)):
                    # print(X_t_arbre.loc[[indices_oob_arbre[i]]])
                    X_t_arbre.set_value(indices_oob_arbre[i],feature, X_t.get_value(permuted_index[i],feature))
                    # print(X_t_arbre.loc[[indices_oob_arbre[i]]])
                    total[indices_oob_arbre[i]]= total[indices_oob_arbre[i]] + 1
                    if forest.sk_learn:
                        vote[indices_oob_arbre[i]] = vote[indices_oob_arbre[i]] + (
                        forest.trees[arbre].predict(X_t_arbre.loc[[indices_oob_arbre[i]]])[0] == Y_t.loc[
                            [indices_oob_arbre[i]]].as_matrix())[0]
                    else:
                        if forest.L == 1:
                            vote[indices_oob_arbre[i]] = vote[indices_oob_arbre[i]] + (
                                forest.trees[arbre].predict( X_t_arbre.loc[indices_oob_arbre[i]].as_matrix()) == Y_t.loc[
                            [indices_oob_arbre[i]]].as_matrix())[0]
                        else:
                            vote[indices_oob_arbre[i]] = vote[indices_oob_arbre[i]] + (
                                forest.trees[arbre].predict(X_t_arbre.loc[indices_oob_arbre[i]].as_matrix()) == Y_t.loc[
                                    [indices_oob_arbre[i]]].as_matrix())[0]
                    # vote[indices_oob_arbre[i]] = vote[indices_oob_arbre[i]] + (forest.trees[arbre].predict(X_t_arbre.loc[[indices_oob_arbre[i]]])[0] == Y_t.loc[[indices_oob_arbre[i]]].as_matrix())[0]
        for x in X_t.index:
            if (vote[x] < int(total[x] / 2)):
                errors[feature] = errors[feature] + 1.
        errors[feature] = errors[feature]/float(len(X_t))

    return errors

def single_tree_error(X_t,Y_t,forest):
    errors = [0]*forest.forest_size
    totals = [0]*forest.forest_size
    for x in range(X_t.shape[0]):
        #We use the majority vote
        for arbre in range(forest.forest_size):
            #test if we consider this tree - is the data x used to construct the tree arbre
            if not X_t.index[x] in forest.forest_indices[arbre]:
                totals[arbre] = totals[arbre] + 1
                if forest.sk_learn:
                    cond = (forest.trees[arbre].predict(X_t.iloc[[x]].as_matrix())[0]!=Y_t.iloc[[x]].as_matrix())
                else:
                    if forest.L == 1:
                        cond = (forest.trees[arbre].predict(X_t.iloc[x].as_matrix())!=Y_t.iloc[x].as_matrix())
                    else:
                        cond = (forest.trees[arbre].predict(X_t.iloc[x].as_matrix()) != Y_t.iloc[
                            x].as_matrix())
                if cond:
                    errors[arbre] = errors[arbre] + 1.
    errors = [errors[i]/totals[i] for i in range(forest.forest_size)]
    return np.mean(errors)

def swap_first_last(data):
    for r in range(len(data)):
        temp = data[r][0]
        data[r][0]=data[r][-1]
        data[r][-1]=temp
    return data

