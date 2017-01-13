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


def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                gini += 1.1
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


class Node:

    def __init__(self):
        self.index= 999
        self.value = 999
        self.score = 999
        self.terminal = False
        self.pred = 0
        self.left = None
        self.right = None


    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, dataset):
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Select the best split point for a dataset
    def get_split(self, dataset, F):
        indexes = random.sample(range(0, len(dataset[0]) - 1), F)
        class_values = list(set(row[-1] for row in dataset))
        for index in indexes:
            for row in dataset:
                left, right = self.test_split(index, row[index], dataset)
                gini = gini_index([left, right], class_values)
                if gini < self.score:
                    self.index, self.value, self.score = index, row[index], gini

        return left, right


    # Create a terminal node value
    def to_terminal(self, group):
        self.terminal = True
        outcomes = [row[-1] for row in group]
        self.pred = max(set(outcomes), key=outcomes.count)


    # Create child splits for a node or make terminal
    def split(self, dataset, max_depth, min_size, depth, F):
        left, right = self.get_split(dataset, F)
        self.left = Node()
        self.right = Node()
        if not left and not right:
            return
        # check for max depth
        if depth >= max_depth:
            if not left or not right:
                if len(right) != 0:
                    self.right.to_terminal(right)
                    self.left.to_terminal(right)
                if len(left) != 0:
                    self.right.to_terminal(left)
                    self.left.to_terminal(left)
            else:
                    self.right.to_terminal(right)
                    self.left.to_terminal(left)
            return
        if not left or not right:
            # process right child, left is empty
            if len(right) != 0:
                if len(right) <= min_size:
                    self.right.to_terminal(right)
                else:
                    self.right.split(right, max_depth, min_size, depth+1, F)
                self.left = self.right
            if len(left) != 0:
                if len(left) <= min_size:
                    self.left.to_terminal(left)
                else:
                    self.left.split(left, max_depth, min_size, depth + 1, F)
                self.right = self.left
        else:
            if len(left) <= min_size:
                self.left.to_terminal(left)
            else:
                self.left.split(left, max_depth, min_size, depth + 1, F)
            # process right child
            if len(right) <= min_size:
                self.right.to_terminal(right)
            else:
                self.right.split(right, max_depth, min_size, depth + 1, F)

            #	# check for a no split
            #	if not left or not right:
            #		node['left'] = node['right'] = to_terminal(left + right)
            #		return
            # check for max depth


    # if depth >= max_depth:
    #		node['left'], node['right'] = to_terminal(left), to_terminal(right)
    #		return


    # Build a decision tree
    # def build_tree(self, train, max_depth, min_size, F):
    #     root = get_split(self, train, F)
    #     split(self, root, max_depth, min_size, 1, F)
    #     return root
    #

    # Make a prediction with a decision tree
    def predict(self, row):
        if self.terminal:
            return self.pred
        else:
            if row[self.index] < self.value:
                return self.left.predict(row)

            else:
                return self.right.predict(row)



    # # Classification and Regression Tree Algorithm
    # def decision_tree(train, test, max_depth, min_size):
    #     tree = build_tree(train, max_depth, min_size)
    #     predictions = list()
    #     for row in test:
    #         prediction = predict(tree, row)
    #         predictions.append(prediction)
    #     return (predictions)
