"""
@author: Julien
"""

import decision_tree
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import csv
import numpy as np
import pandas
import warnings
import matplotlib.pyplot as plt
from Forest_RI_2 import *
warnings.filterwarnings("ignore", category=DeprecationWarning)

ITER_NUMBER = 1
FOREST_SIZE = 100
SK_LEARN = False
F = [1]

test_set_error = [[], []]
error_selection_tab = []
generalisation_error_selection = []
generalisation_error_single = []
generalisation_error_one_tree = []
adaboost_error = 0

data_file = "datasets/pima-indians-diabetes.txt"  # "datasets/ionosphere.txt"######

f = open(data_file, "r")

X = []
with open(data_file, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        X.append(row)

X = convert_to_float(X)
df = pandas.DataFrame(X)


for iter in range(ITER_NUMBER):
    out_of_bag = []
    # Take all dataframe
    df_train = df
    Y_t = df_train.drop(df_train.columns[:len(df.T) - 1], axis=1)
    X_t = df_train.drop(df_train.columns[len(df.T) - 1], axis=1)

    f_forests = []

    for f in range(len(F)):
        forest = Forest(FOREST_SIZE, SK_LEARN, F[f])
        forest.train(df_train)

        f_forests.append(forest)

        # we compute the out_of_bag error in the forest
        out_of_bag.append(out_of_bag_error(X_t, Y_t, forest))

        print(out_of_bag[0])
        perm = [(oob_p - out_of_bag[0]) * 100 / out_of_bag[0] for oob_p in permuted_oob_error(X_t, Y_t, forest)]
        plt.scatter(range(len(perm)), perm)
        plt.show()
