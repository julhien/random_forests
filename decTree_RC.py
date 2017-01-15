# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 15:31:34 2017

@author: Mathilde
"""

from random import seed
from random import randrange
from csv import reader
import numpy as np
import random

def fraction(group, output):
    return [row[-1] for row in group].count(output) / float(len(group))

def gini_score(groups, classes):
    groups = [group for group in groups if len(group)>0]
    return sum(sum(fraction(group, output)*(1.0 - fraction(group, output)) for group in groups)for output in classes)

class Node:
    def __init__(self):
        self.features= None
        self.weights = None
        self.value = None
        self.classes = []
        self.gini = 1000
        self.terminal = False
        self.pred = 0
        self.left = None
        self.right = None


    def split_at(self, indices, weights, value, dataset):
        left = [row for row in dataset if sum([weights[i]*row[indices[i]] for i in range(len(indices))]) < value]
        right = [row for row in dataset if sum([weights[i]*row[indices[i]] for i in range(len(indices))]) >= value]

        return left, right


    def find_split(self, dataset, F, L):
        self.classes = list(set(row[-1] for row in dataset))
        l_candidate = None
        r_candidate = None
        nb_features = len(dataset[0]) - 1
        weights = np.random.rand(L, F) * 2 - 1
        candidates = []
        for f in range(F):
            candidates.append(random.sample(range(nb_features), L))
        candidates = np.array(candidates).T
        limits = np.zeros((2, F))
        for f in range(F):
            for l in range(L):
                a = np.array(dataset)[:, candidates[l, f]].astype(float) * weights[l, f]
                limits[0, f] += np.min(a)
                limits[1, f] += np.max(a)
        for f in range(F):
            for value in np.linspace(limits[0, f], limits[1, f], 50):
                left, right = self.split_at(candidates[:, f], weights[:, f], value, dataset)
                gini = gini_score([left, right], self.classes)
                if gini < self.gini:
                    l_candidate = left
                    r_candidate = right
                    self.features, self.weights, self.value, self.gini = candidates[:, f], weights[:, f], value, gini

        return l_candidate, r_candidate


    def leaf(self, group):
        self.terminal = True
        classes = [row[-1] for row in group]
        self.pred = max(set(classes), key=classes.count)



    def split(self, left, right, max_depth, min_size, depth, F, L):
        self.left = Node()
        self.right = Node()

        if not left and not right:
            return

        if depth >= max_depth:
            if not left or not right:
                if len(right) != 0:
                    self.right.leaf(right)
                    self.left.leaf(right)
                if len(left) != 0:
                    self.right.leaf(left)
                    self.left.leaf(left)
            else:
                    self.right.leaf(right)
                    self.left.leaf(left)
            return
        if not left or not right:
            if len(right) != 0:
                if len(right) <= min_size:
                    self.right.leaf(right)
                else:
                    l_candidate, r_candidate = self.right.find_split(right, F, L)
                    if self.gini - self.right.gini>0.001:
                    # if True:
                        self.right.split(l_candidate, r_candidate, max_depth, min_size, depth+1, F, L)
                    else:
                        self.right.leaf(right)
                self.left.leaf(right)
            if len(left) != 0:
                if len(left) <= min_size:
                    self.left.leaf(left)
                else:
                    l_candidate, r_candidate = self.left.find_split(left, F, L)
                    if self.gini - self.left.gini>0.001:
                    # if True:
                        self.left.split(l_candidate, r_candidate, max_depth, min_size, depth+1, F, L)
                    else:
                        self.left.leaf(left)
                self.right.leaf(left)
        else:
            if len(left) <= min_size:
                self.left.leaf(left)
            else:
                l_candidate, r_candidate = self.left.find_split(left, F, L)
                self.left.split(l_candidate, r_candidate, max_depth, min_size, depth + 1, F, L)

            if len(right) <= min_size:
                self.right.leaf(right)
            else:
                l_candidate, r_candidate = self.right.find_split(right, F, L)
                self.right.split(l_candidate, r_candidate, max_depth, min_size, depth + 1, F, L)

    def train(self, dataset, max_depth, min_size, F, L):
        left, right = self.find_split(dataset, F, L)
        self.split(left, right, max_depth, min_size, 1, F, L)


    def predict(self, x):
        if self.terminal:
            return self.pred
        else:
            if sum([self.weights[i]*x[self.features[i]] for i in range(len(self.features))]) < self.value:
                return self.left.predict(x)

            else:
                return self.right.predict(x)

