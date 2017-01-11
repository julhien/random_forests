# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 17:44:13 2017

@author: Mathilde
"""
import decision_tree_bis
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import csv
import numpy as np
import pandas
import glob
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



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
        for arbre in range(len(forest)):
            #vote = vote + (forest[arbre].predict(X_test.iloc[[x]].as_matrix())[0]==Y_test.iloc[[x]].as_matrix())
            vote = vote + (decision_tree_bis.predict(forest[arbre],X_test.iloc[x].as_matrix())==Y_test.iloc[x].as_matrix())
        if (vote <int(len(forest)/2)):
            error = error + 1.
    return error/len(X_test)
    
def out_of_bag_error(X_t,Y_t,forest,forest_indexes):
    error = 0.
    for x in range(X_t.shape[0]):
        #We use the majority vote
        vote = 0
        total = 0
        for arbre in range(len(forest)):
            #test if we consider this tree - is the data x used to construct the tree arbre
            if not X_t.index[x] in forest_indexes[arbre]:
                total = total + 1
                #vote = vote + (forest[arbre].predict(X_t.iloc[[x]].as_matrix())[0]==Y_t.iloc[[x]].as_matrix())
                vote = vote + (decision_tree_bis.predict(forest[arbre],X_test.iloc[x].as_matrix())==Y_test.iloc[x].as_matrix())
        if (vote <int(total/2)):
            error = error + 1.

    return error/len(X_t)

def single_tree_error(X_t,Y_t,forest,forest_indexes):
    errors = [0]*len(forest)
    totals = [0]*len(forest)
    for x in range(X_t.shape[0]):
        #We use the majority vote
        for arbre in range(len(forest)):
            #test if we consider this tree - is the data x used to construct the tree arbre
            if not X_t.index[x] in forest_indexes[arbre]:
                totals[arbre] = totals[arbre] + 1
                #if (forest[arbre].predict(X_t.iloc[[x]].as_matrix())[0]!=Y_t.iloc[[x]].as_matrix()):
                if (decision_tree_bis.predict(forest[arbre],X_t.iloc[x].as_matrix())!=Y_t.iloc[x].as_matrix()):
                    errors[arbre] = errors[arbre] + 1.
    errors = [errors[i]/totals[i] for i in range(len(forest))]
    return np.mean(errors)

def swap_first_last(data):
    for r in range(len(data)):
        temp = data[r][0]
        data[r][0]=data[r][-1]
        data[r][-1]=temp
    return data
   
file = open("result\\results_Forest_RI.txt", "w")
for data_file in glob.glob("datasets/*.txt"):
    print data_file
    f = open( data_file, "r" )

    X = []
    
    with open(data_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            X.append(row)
    to_swap = ['datasets\\votes.txt','datasets\\image.txt','datasets\\soybean.txt']
    if data_file in to_swap: 
        X = swap_first_last(X)
    X = convert_to_float(X)
    
    
    file.write(data_file[9:-4]+'\n')
    
    df = pandas.DataFrame(X)
    #Number of feature selected at each node
    F = [1,int(np.log(len(df.T)-1)/np.log(2)-1)]
    test_set_error = [[],[]]
    error_selection_tab = []
    generalisation_error_selection = []
    generalisation_error_single = []
    generalisation_error_one_tree = []
    adaboost_error = 0
    for iter in range(100):
        out_of_bag = []
        # Randomly sample 90% of the dataframe
        df_train = df.sample(frac=0.9)
    
        df_test = df.loc[~df.index.isin(df_train.index)]
        
        Y_t = df_train.drop(df_train.columns[:len(df.T)-1], axis=1)
    
        X_t = df_train.drop(df_train.columns[len(df.T)-1], axis=1)
        Y_test = df_test.drop(df_test.columns[:len(df.T)-1], axis=1)
        X_test = df_test.drop(df_test.columns[len(df.T)-1], axis=1)
        
        #Adaboost with 50 trees
        adaboost = AdaBoostClassifier()
        adaboost = adaboost.fit(X_t, Y_t)
        adaboost_error = adaboost_error + adaboost.score(X_test,Y_test)
    
        f_forests=[]
        f_forests_indices = []
        for f in range(len(F)):
            forest = []
            #we keep track of the index of the data used for each tree
            forest_indexes = []
            oob_error=0
            # For each tree of the forest
            for arbre in range(100):
                #bagging
                df_train_bagged = df_train.sample(frac=0.66,replace=True)
                forest_indexes.append(df_train_bagged.index)
                ##Construct a forest thanks to sklearn
    #            Y_train = df_train_bagged.drop(df_train_bagged.columns[:len(df.T)-1], axis=1)
    #            X_train = df_train_bagged.drop(df_train_bagged.columns[len(df.T)-1], axis=1)
    #            clf = tree.DecisionTreeClassifier(max_features = F[f])
    #            forest.append(clf.fit(X_train, Y_train))
                
                # construct a forest with decision tree
                forest.append(decision_tree_bis.build_tree(df_train_bagged.values.tolist(),150,5,F[f]))
                
                #out_of_bag estimate
                total = 0.
                vote = 0.
                for x in range(len(X_t)):
                    if not X_t.index[x] in forest_indexes[-1]:
                        total = total + 1.
                        vote = vote + int(decision_tree_bis.predict(forest[-1],X_t.iloc[x])!=Y_t.iloc[x])
                oob_error = oob_error + vote/total
                    
                
            f_forests.append(forest)
            f_forests_indices.append(forest_indexes)
            # ... now we are supposed to have a beautiful forest
                
            #we compute the out_of_bag error in the forest
            #out_of_bag.append(out_of_bag_error(X_t,Y_t,forest,forest_indexes)) 
            out_of_bag.append(oob_error/100.) 
            
            #we compute the test set error on the forest
            test_set_error[f].append(function_test_set_error(X_test,Y_test,forest))
        #select the test error from the run which has the lower out_of_bag estimate
        error_selection_tab.append(test_set_error[np.argmin(out_of_bag)][-1])
        #select the lower of the out_of_bag estimates
        generalisation_error_selection.append(np.min(out_of_bag))
        #out_of_bag estimate for single setting
        generalisation_error_single.append(out_of_bag[0])
        #evaluating the best estimator with regards to test set
        index_best  = np.argmin([test_set_error[i][-1] for i in range(len(F))])
        #calculate individual trees generalization errors wrt this best
        generalisation_error_one_tree.append(
            single_tree_error(X_t, Y_t, f_forests[index_best], f_forests_indices[index_best]))
    
    adaboost_error = adaboost_error/100.
    error_selection = np.mean(error_selection_tab)
    error_single = np.mean(test_set_error[0])
    error_one_tree = np.mean(generalisation_error_one_tree)
    #print(error_selection_tab)
    #☼print(test_set_error)
    file.write('Adaboost = '+str(adaboost_error)+'\n'+'Selection = '+str(error_selection)+'\n'+'Single = '+str(error_single)+'\n'+'One Tree = '+str(error_one_tree)+'\n\n')
    
#    print(adaboost_error)
#    print(error_selection)
#    print(error_single)
#    print(error_one_tree)
file.close()
# We need to compute the generalis ation error for the best setting between single and selection
one_tree = 0
#
# if(error_selection<=error_single):
#     #average the out_of_bag error, for each iteration we take the lower between selection and single
#     one_tree = np.mean(generalisation_error_selection)
# else:
#     one_tree = np.mean(generalisation_error_single)
#