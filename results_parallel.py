# -*- coding: utf-8 -*-
"""
@author: Julien
"""
import decision_tree_bis
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import csv
import numpy as np
import pandas
import glob
import warnings
from multiprocessing import Pool
from Forest_RI_2 import *

warnings.filterwarnings("ignore", category=DeprecationWarning)

SK_LEARN = False
FOREST_SIZE = 100
L = 1
NUMBER_ITER = 1


def forest_grow(df):
    F = [1, int(np.log(len(df.T) - 1) / np.log(2) - 1)]
    out_of_bag = []
    test_set_error = []
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
    adaboost_error = 1 - adaboost.score(X_test, Y_test)

    f_forests = []

    for f in range(len(F)):
        print("next F")
        forest = Forest(FOREST_SIZE, SK_LEARN, F[f], L)
        forest.train(df_train)

        f_forests.append(forest)
        # ... now we are supposed to have a beautiful forest

        # we compute the out_of_bag error in the forest
        out_of_bag.append(out_of_bag_error(X_t, Y_t, forest))
        # out_of_bag.append(oob_error / 100.)

        # we compute the test set error on the forest
        test_set_error.append(function_test_set_error(X_test, Y_test, forest))

    # select the test error from the run which has the lower out_of_bag estimate
    error_selection = test_set_error[np.argmin(out_of_bag)]
    error_single = test_set_error[0]
    # select the lower of the out_of_bag estimates
    generalisation_error_selection = np.min(out_of_bag)
    # out_of_bag estimate for single setting
    generalisation_error_single = out_of_bag[0]
    # evaluating the best estimator with regards to test set
    index_best = np.argmin([test_set_error[i] for i in range(len(F))])
    # calculate individual trees generalization errors wrt this best
    generalisation_error_one_tree = single_tree_error(X_t, Y_t, f_forests[index_best])

    return [adaboost_error, error_selection, error_single, generalisation_error_one_tree]

for data_file in ["datasets/votes.txt", "datasets/vehicle.txt", "datasets/ecoli.txt"]:
# for data_file in [
#                  "datasets/image.txt","datasets/pima-indians-diabetes.txt","datasets/ionosphere.txt","datasets/sonar.txt","datasets/vehicle.txt",
#                   "datasets/votes.txt","datasets/vowel.txt", "datasets/ecoli.txt","datasets/glass.txt", "datasets/german.txt"]:
    file = open("result/results_test/results_Forest_RC_par_"+data_file[9:], "w")
# file = open("result\\results_Forest_RI_ionosphere.txt", "w")
# for data_file in ["datasets/sonar.txt"]:
    print data_file
    f = open(data_file, "r")

    X = []

    with open(data_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            X.append(row)
    to_swap = ['datasets/votes.txt', 'datasets/image.txt']

    if data_file in to_swap:
        X = swap_first_last(X)
    X = convert_to_float(X)


    file.write(data_file[9:-4] + '\n')

    df = pandas.DataFrame(X)
    if data_file in ["datasets/glass.txt","datasets/ecoli.txt"]:
        df = df.drop(df.columns[0], axis=1)
        print(df)
    if data_file in ["datasets/vowel.txt"]:
        df = df.drop(df.columns[:3], axis=1)
    # Number of feature selected at each node
    #F = [1, int(np.log(len(df.T) - 1) / np.log(2) - 1)]
    print(df)
    # forest_grow(df)
    pool = Pool(processes=6)
    y_parallel = pool.map(forest_grow, [df for i in range(NUMBER_ITER)])
    pool.close()
    pool.join()
    y_parallel = np.array(y_parallel)

    # print(y_parallel)
    adaboost_error = np.mean(y_parallel[:,0])
    error_selection = np.mean(y_parallel[:,1])
    error_single = np.mean(y_parallel[:,2])
    error_one_tree = np.mean(y_parallel[:,3])
    # print(error_selection_tab)
    # ☼print(test_set_error)
    file.write(
        'Adaboost = ' + str(adaboost_error) + '\n' + 'Selection = ' + str(error_selection) + '\n' + 'Single = ' + str(
            error_single) + '\n' + 'One Tree = ' + str(error_one_tree) + '\n\n')
    print(
        'Adaboost = ' + str(adaboost_error) + '\n' + 'Selection = ' + str(error_selection) + '\n' + 'Single = ' + str(
            error_single) + '\n' + 'One Tree = ' + str(error_one_tree) + '\n\n')


    file.close()
