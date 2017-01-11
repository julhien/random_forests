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

warnings.filterwarnings("ignore", category=DeprecationWarning)

SK_LEARN = True
FOREST_SIZE = 40
NUMBER_ITER = 2


# file = open("result\\results_Forest_RI.txt", "w")
# for data_file in glob.glob("datasets/*.txt"):

file = open("result\\results_Forest_RI_ionosphere.txt", "w")
for data_file in ["datasets/ionosphere.txt"]:
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
    F = [1, int(np.log(len(df.T) - 1) / np.log(2) - 1)]
    test_set_error = [[],[]]  # test-set error for forests grown all Fs
    error_selection_tab = []  # test-set error for forest selected as best test-error performance
    generalisation_error_selection = []  # error of out-of-bag estimate for forest selected as best test-error performance
    generalisation_error_single = []  # error of out-of-bag estimate for forest grown with F=1
    generalisation_error_one_tree = []  # error of out-of-bag estimate for each individual tree of the selected forest
    adaboost_error = 0

    for iter in range(NUMBER_ITER):
        out_of_bag = []

        # Randomly sample 90% of the dataframe
        df_train = df.sample(frac=0.9)
        df_test = df.loc[~df.index.isin(df_train.index)]

        Y_t = df_train.drop(df_train.columns[:len(df.T) - 1], axis=1)
        X_t = df_train.drop(df_train.columns[len(df.T) - 1], axis=1)
        Y_test = df_test.drop(df_test.columns[:len(df.T) - 1], axis=1)
        X_test = df_test.drop(df_test.columns[len(df.T) - 1], axis=1)

        # Adaboost with 50 trees
        adaboost = AdaBoostClassifier()
        adaboost = adaboost.fit(X_t, Y_t)
        adaboost_error = adaboost_error + adaboost.score(X_test, Y_test)

        f_forests = []

        for f in range(len(F)):
            forest = Forest_RI(FOREST_SIZE, SK_LEARN, F[f])
            forest.train(df_train)
            # forest = []
            # # we keep track of the index of the data used for each tree
            # forest_indexes = []
            # oob_error = 0
            # # For each tree of the forest
            # for arbre in range(FOREST_SIZE):
            #     # bagging
            #     df_train_bagged = df_train.sample(frac=1., replace=True)
            #     forest_indexes.append(df_train_bagged.index)
            #     if SK_LEARN:
            #         ##Construct a forest thanks to sklearn
            #         Y_train = df_train_bagged.drop(df_train_bagged.columns[:len(df.T) - 1], axis=1)
            #         X_train = df_train_bagged.drop(df_train_bagged.columns[len(df.T) - 1], axis=1)
            #         clf = tree.DecisionTreeClassifier(max_features=F[f])
            #         forest.append(clf.fit(X_train, Y_train))
            #
            #     # construct a forest with decision tree
            #     else:
            #         forest.append(decision_tree_bis.build_tree(df_train_bagged.values.tolist(), 150, 5, F[f]))

                # # out_of_bag estimate
                # total = 0.
                # vote = 0.
                # for x in range(len(X_t)):
                #     if not X_t.index[x] in forest_indexes[-1]:
                #         total = total + 1.
                #         vote = vote + int(decision_tree_bis.predict(forest[-1], X_t.iloc[x]) != Y_t.iloc[x])
                # oob_error = oob_error + vote / total

            f_forests.append(forest)
            # ... now we are supposed to have a beautiful forest

            # we compute the out_of_bag error in the forest
            out_of_bag.append(out_of_bag_error(X_t,Y_t,forest))
            # out_of_bag.append(oob_error / 100.)

            # we compute the test set error on the forest
            test_set_error[f].append(function_test_set_error(X_test, Y_test, forest))

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
