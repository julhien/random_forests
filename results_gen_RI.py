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
from Forest_RI_2 import *
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

SK_LEARN = True
FOREST_SIZE = 100
L = 1
NUMBER_ITER = 1


# file = open("result\\results_Forest_RI.txt", "w")
# for data_file in glob.glob("datasets/*.txt"):

file = open("result\\results_Forest_RI.txt", "w")
list_dataset_RI = glob.glob("datasets/*.txt")
list_dataset_RI.remove('datasets\\soybean.txt')
list_dataset_RI.remove('datasets\\ecoli.txt')
#for data_file in list_dataset_RI:
for data_file in ['datasets\\sonar.txt']:
    print data_file
    f = open(data_file, "r")

    X = []

    with open(data_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            X.append(row)
    to_swap = ['datasets\\votes.txt', 'datasets\\image.txt', 'datasets\\soybean.txt']
    if data_file in to_swap:
        X = swap_first_last(X)
    X = convert_to_float(X)

    file.write(data_file[9:-4] + '\n')

    df = pandas.DataFrame(X)
    # Number of feature selected at each node
    #F=[1]
    #F = [1, int(np.log(len(df.T) - 1) / np.log(2) - 1)]
    F = range(1,10)
    test_set_error = [[],[]]  # test-set error for forests grown all Fs
    error_selection_tab = []  # test-set error for forest selected as best test-error performance
    generalisation_error_selection = []  # error of out-of-bag estimate for forest selected as best test-error performance
    generalisation_error_single = []  # error of out-of-bag estimate for forest grown with F=1
    generalisation_error_one_tree = []  # error of out-of-bag estimate for each individual tree of the selected forest
    generalisation_out_of_bag_strength = []
    generalisation_out_of_bag_cor = []
    adaboost_error = 0

    for iter in range(NUMBER_ITER):
        out_of_bag = []
        out_of_bag_strength = []
        out_of_bag_cor = []
        # Randomly sample 90% of the dataframe
        df_train = df.sample(frac=0.9)
        df_test = df.loc[~df.index.isin(df_train.index)]
        ##Normalize for RC
        df_train, mean, std = normalize_train(df_train)
        Y_t = df_train.drop(df_train.columns[:len(df.T) - 1], axis=1)
        X_t = df_train.drop(df_train.columns[len(df.T) - 1], axis=1)
        Y_test = df_test.drop(df_test.columns[:len(df.T) - 1], axis=1)
        X_test = df_test.drop(df_test.columns[len(df.T) - 1], axis=1)
        X_test = normalize_test(X_test, mean, std)
        # Adaboost with 50 trees
        adaboost = AdaBoostClassifier()
        adaboost = adaboost.fit(X_t, Y_t)
        adaboost_error = adaboost_error + adaboost.score(X_test, Y_test)

        f_forests = []

        for f in range(len(F)):
            forest = Forest(FOREST_SIZE, SK_LEARN, F[f], L)
            forest.train(df_train)

            f_forests.append(forest)

            # we compute the out_of_bag error in the forest
            out_of_bag.append(out_of_bag_error(X_t,Y_t,forest))
            # out_of_bag.append(oob_error / 100.)
            aux = out_of_bag_str(X_t,Y_t,forest)
            out_of_bag_strength.append(aux[0])
            out_of_bag_cor.append(aux[1])
            # we compute the test set error on the forest
            #test_set_error[f].append(function_test_set_error(X_test, Y_test, forest))
            
        
        generalisation_out_of_bag_strength.append(out_of_bag_strength)
        strength = np.mean(generalisation_out_of_bag_strength,axis=0)
        generalisation_out_of_bag_cor.append(out_of_bag_cor)
        cor = np.mean(generalisation_out_of_bag_cor,axis=0)
        plt.plot(F,strength,'o')
        plt.plot(F,cor,'x')
        plt.show()
        # select the test error from the run which has the lower out_of_bag estimate
        error_selection_tab.append(test_set_error[np.argmin(out_of_bag)][-1])
        # select the lower of the out_of_bag estimates
        generalisation_error_selection.append(np.min(out_of_bag))
        # out_of_bag estimate for single setting
        generalisation_error_single.append(out_of_bag[0])
        # evaluating the best estimator with regards to test set
        index_best = np.argmin([test_set_error[i][-1] for i in range(len(F))])
        # calculate individual trees generalization errors wrt this best
        generalisation_error_one_tree.append(
            single_tree_error(X_t, Y_t, f_forests[index_best]))

    adaboost_error = adaboost_error / 100.
    error_selection = np.mean(error_selection_tab)
    error_single = np.mean(test_set_error[0])
    error_one_tree = np.mean(generalisation_error_one_tree)
    # print(error_selection_tab)
    # ☼print(test_set_error)
    file.write(
        'Adaboost = ' + str(adaboost_error) + '\n' + 'Selection = ' + str(error_selection) + '\n' + 'Single = ' + str(
            error_single) + '\n' + 'One Tree = ' + str(error_one_tree) + '\n\n')
    print(
        'Adaboost = ' + str(adaboost_error) + '\n' + 'Selection = ' + str(error_selection) + '\n' + 'Single = ' + str(
            error_single) + '\n' + 'One Tree = ' + str(error_one_tree) + '\n\n')

file.close()
