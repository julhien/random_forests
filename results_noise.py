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
import random
from multiprocessing import Pool
from Forest_RI_2 import *

warnings.filterwarnings("ignore", category=DeprecationWarning)

SK_LEARN = False
FOREST_SIZE = 100
NUMBER_ITER = 24

def swap_classes(df):
    indices_swap = np.random.choice(range(len(df.index)),size = len(df.index)/20 , replace=False)
    before = df.copy()
    classes = list(set(df.iloc[:,-1].values.tolist()))
    for index in indices_swap:
        df.set_value(df.index[index], df.columns[-1], np.random.choice([c for c in classes if c!=df.iloc[index,-1]],size=1)[0])
    # print(sum([df.iloc[i,-1] != before.iloc[i,-1] for i in range(len(df.index))])/(float)(len(df.index)))
    return df
def forest_grow(df):
    F = [2,4]
    L = [3,1]
    test_set_error_RC = []
    test_set_error_RI = []
    test_set_error_Adaboost = []
    # Randomly sample 90% of the dataframe
    df_train = df.sample(frac=0.9)
    df_test = df.loc[~df.index.isin(df_train.index)]
    ##Normalize for RC
    df_train, mean, std = normalize_train(df_train)
    Y_test = df_test.drop(df_test.columns[:len(df.T) - 1], axis=1)
    X_test = df_test.drop(df_test.columns[len(df.T) - 1], axis=1)
    X_test = normalize_test(X_test, mean, std)

    f_forests = []

    for swap_state in [False, True]:
        print("next F")
        if swap_state:
            df_train = swap_classes(df_train)

        Y_t = df_train.drop(df_train.columns[:len(df.T) - 1], axis=1)
        X_t = df_train.drop(df_train.columns[len(df.T) - 1], axis=1)

        adaboost = AdaBoostClassifier()
        adaboost = adaboost.fit(X_t, Y_t)
        test_set_error_Adaboost.append(1 - adaboost.score(X_test, Y_test))

        forest_RC = Forest(FOREST_SIZE, SK_LEARN, F[0], L[0])
        forest_RC.train(df_train)

        # we compute the test set error on the forest
        test_set_error_RC.append(function_test_set_error(X_test, Y_test, forest_RC))

        forest_ri = Forest(FOREST_SIZE, SK_LEARN, F[1], L[1])
        forest_ri.train(df_train)

        # we compute the test set error on the forest
        test_set_error_RI.append(function_test_set_error(X_test, Y_test, forest_ri))

    diff_Adaboost = (test_set_error_Adaboost[1]-test_set_error_Adaboost[0])*100
    diff_RC = (test_set_error_RC[1]-test_set_error_RC[0])*100
    diff_RI = (test_set_error_RI[1]-test_set_error_RI[0])*100

    return [test_set_error_RC[0], test_set_error_RC[1],
            test_set_error_RI[0], test_set_error_RI[1],
            test_set_error_Adaboost[0], test_set_error_Adaboost[1]]

for data_file in ["datasets/ionosphere.txt","datasets/sonar.txt", "datasets/pima-indians-diabetes.txt", "datasets/votes.txt","datasets/liver.txt", "datasets/vehicle.txt"]:
# for data_file in [
#                  "datasets/image.txt","datasets/pima-indians-diabetes.txt","datasets/ionosphere.txt","datasets/sonar.txt","datasets/vehicle.txt",
#                   "datasets/votes.txt","datasets/vowel.txt", "datasets/ecoli.txt","datasets/glass.txt", "datasets/german.txt"]:
    file = open("result/results_Noise/results_Forest_RI_par_"+data_file[9:], "w")
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
        # print(df)
    if data_file in ["datasets/vowel.txt"]:
        df = df.drop(df.columns[:3], axis=1)
    #
    #
    # print(df)
    pool = Pool(processes=6)
    # forest_grow(df)
    y_parallel = pool.map(forest_grow, [df for i in range(NUMBER_ITER)])
    pool.close()
    pool.join()
    y_parallel = np.array(y_parallel)

    # print(y_parallel)
    diff_adaboost = np.mean(y_parallel[:,0])
    diff_RC = np.mean(y_parallel[:,1])
    diff_RI = np.mean(y_parallel[:,2])
    test_set_error_RC_no = np.mean(y_parallel[:,0])
    test_set_error_RC_yes = np.mean(y_parallel[:,1])
    test_set_error_RI_no = np.mean(y_parallel[:,2])
    test_set_error_RI_yes = np.mean(y_parallel[:,3])
    test_set_error_Adaboost_no = np.mean(y_parallel[:,4])
    test_set_error_Adaboost_yes = np.mean(y_parallel[:,5])

    file.write(
        'Adaboost = ' + str(test_set_error_Adaboost_no)+ ","+str(test_set_error_Adaboost_yes) + '\n' + 'RI = ' + str(test_set_error_RI_no)+","+ str(test_set_error_RI_yes)+ '\n' + 'RC = ' + str(
            test_set_error_RC_no)+","+str(test_set_error_RC_yes)+'\n\n')

    print(
        'Adaboost = ' + str(test_set_error_Adaboost_no)+ ","+str(test_set_error_Adaboost_yes) + '\n' + 'RI = ' + str(test_set_error_RI_no)+","+ str(test_set_error_RI_yes)+ '\n' + 'RC = ' + str(
            test_set_error_RC_no)+","+str(test_set_error_RC_yes)+'\n\n')
    file.close()
