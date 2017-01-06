# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 17:44:13 2017

@author: Mathilde
"""
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import csv
import numpy as np
import pandas
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data_file  =  "sonar.txt"#""ionosphere.txt"

f = open( data_file, "r" )

X = []

with open(data_file, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        X.append(row)


df = pandas.DataFrame(X)

def function_test_set_error(X_test,Y_test,forest):
    error = 0.
    for x in range(len(X_test)):
        #We use the majority vote
        vote = 0
        for arbre in range(len(forest)):
            vote = vote + (forest[arbre].predict(X_test.iloc[x])[0]==Y_test.iloc[x])
        if (vote <int(len(forest)/2)):
            error = error + 1.
    return error/len(X_test)
    
def out_of_bag_error(X_t,Y_t,forest,forest_indexes):
    error = 0.
    for x in range(len(X_t)):
        #We use the majority vote
        vote = 0
        total = 0
        for arbre in range(len(forest)):
            #test if we consider this tree - is the data x used to construct the tree arbre
            if not X_t.index[x] in forest_indexes[arbre]:
                total = total + 1
                vote = vote + (forest[arbre].predict(X_t.iloc[x])[0]==Y_t.iloc[x])
        if (vote <int(total/2)):
            error = error + 1.
    return error/len(X_t)
    
#Number of feature selected at each node
F = [1,int(np.log(len(df.T)-1)/np.log(2)-1)]
test_set_error = [[],[]]
error_selection_tab = []
generalisation_error_selection = []
generalisation_error_single = []
adaboost_error = 0
for iter in range(10):
    out_of_bag = []
    # Randomly sample 90% of the dataframe
    df_train = df.sample(frac=0.9)
    df_test = df.loc[~df.index.isin(df_train.index)]
    
    Y_t = df_train[:][len(df.T)-1]
    X_t = df_train.drop(df_train.columns[len(df.T)-1], axis=1)    
    
    Y_test = df_test[:][len(df.T)-1]
    X_test = df_test.drop(df_test.columns[len(df.T)-1], axis=1)
    
    #Adaboost with 50 trees
    adaboost = AdaBoostClassifier()
    adaboost = adaboost.fit(X_t, Y_t)
    adaboost_error = adaboost_error + adaboost.score(X_test,Y_test)
    
    for f in range(len(F)):
        forest = []
        #we keep track of the index of the data used for each tree
        forest_indexes = []
        # For each tree of the forest
        for arbre in range(100):
            #bagging     
            df_train_bagged = df_train.sample(frac=1.,replace=True)
            Y_train = df_train_bagged[:][len(df.T)-1]
            X_train = df_train_bagged.drop(df_train_bagged.columns[len(df.T)-1], axis=1)
            #
            # We use the tree from sklearn but we ought to code it from scratch... growing and combining the trees        
            clf = tree.DecisionTreeClassifier(max_features = F[f])
            forest.append(clf.fit(X_train, Y_train))
            # ...
            # ...
            # ...              
            forest_indexes.append(df_train_bagged.index)
            
        # ... now we are supposed to have a beautiful forest
            
        #we compute the out_of_bag error in the forest
        out_of_bag.append(out_of_bag_error(X_t,Y_t,forest,forest_indexes)) 
        
        #we compute the test set error on the forest
        test_set_error[f].append(function_test_set_error(X_test,Y_test,forest))
    print out_of_bag
    #select the test error from the run which has the lower out_of_bag estimate
    error_selection_tab.append(test_set_error[np.argmin(out_of_bag)][-1])
    #select the lower of the out_of_bag estimates
    generalisation_error_selection.append(np.min(out_of_bag))
    #out_of_bag estimate for single setting
    generalisation_error_single.append(out_of_bag[0])
    
adaboost_error = adaboost_error/100.
error_selection = np.mean(error_selection_tab)
error_single = np.mean(test_set_error[0])

# We need to compute the generalisation error for the best setting between single and selection
one_tree = 0
if(error_selection<=error_single):
    #average the out_of_bag error, for each iteration we take the lower between selection and single
    one_tree = np.mean(generalisation_error_selection)
else:
    one_tree = np.mean(generalisation_error_single)
    