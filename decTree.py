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
        self.feature= None
        self.threshold = None
        self.classes = []
        self.gini = 1000
        self.terminal = False
        self.pred = 0
        self.left = None
        self.right = None


    def split_at(self, index, threshold, data):
        left = [row for row in data if row[index] < threshold]
        right = [row for row in data if row[index] >= threshold]

        return left, right


    def find_split(self, data, F):
        self.classes = list(set(row[-1] for row in data))
        l_candidate = None
        r_candidate = None
        nb_features = len(data[0]) - 1
        indices = range(nb_features)
        empty = True
        while empty and indices:
            candidates = random.sample(indices, F)

            for feature in candidates:
                indices.remove(feature)
                for row in data:
                    left, right = self.split_at(feature, row[feature], data)
                    gini = gini_score([left, right], self.classes)
                    if gini < self.gini:
                        l_candidate = left
                        r_candidate = right
                        self.feature, self.threshold, self.gini = feature, row[feature], gini
            # empty = (len(l_candidate)==0 or len(r_candidate)==0)
            empty = False
        return l_candidate, r_candidate


    def leaf(self, group):
        self.terminal = True
        classes = [row[-1] for row in group]
        self.pred = max(set(classes), key=classes.count)



    def split(self, left, right, max_depth, min_size, depth, F):
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
                    l_candidate, r_candidate = self.right.find_split(right, F)
                    if self.gini - gini_score([l_candidate, r_candidate], self.classes)>0.001:
                        self.right.split(l_candidate, r_candidate, max_depth, min_size, depth+1, F)
                    else:
                        self.right.leaf(right)
                self.left.leaf(right)
            if len(left) != 0:
                if len(left) <= min_size:
                    self.left.leaf(left)
                else:
                    l_candidate, r_candidate = self.left.find_split(left, F)
                    if self.gini - gini_score([l_candidate, r_candidate], self.classes)>0.001:
                        self.left.split(l_candidate, r_candidate, max_depth, min_size, depth+1, F)
                    else:
                        self.left.leaf(left)
                self.right.leaf(left)
        else:
            if len(left) <= min_size:
                self.left.leaf(left)
            else:
                l_candidate, r_candidate = self.left.find_split(left, F)
                self.left.split(l_candidate, r_candidate, max_depth, min_size, depth + 1, F)

            if len(right) <= min_size:
                self.right.leaf(right)
            else:
                l_candidate, r_candidate = self.right.find_split(right, F)
                self.right.split(l_candidate, r_candidate, max_depth, min_size, depth + 1, F)

    def train(self, dataset, max_depth, min_size, F):
        left, right = self.find_split(dataset, F)
        self.split(left, right, max_depth, min_size, 1, F)


    def predict(self, x):
        if self.terminal:
            return self.pred
        else:
            if x[self.feature] < self.threshold:
                return self.left.predict(x)

            else:
                return self.right.predict(x)

