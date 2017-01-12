# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 15:31:34 2017

@author: Julien
"""

from random import seed
from random import randrange
from csv import reader
import numpy as np
import random

# Load a CSV file
def load_csv(filename):
	file = open(filename, "rb")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

## Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Split a dataset based on an attribute and an attribute value
def test_split(index, w, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if sum([w[i]*row[index[i]] for i in range(len(index))]) < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0 :
                gini += 1.1
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini

# Select the best split point for a dataset
def get_split(dataset, F, L):
	if L == 1:
		w = np.array([[1]] * F)
	else:
		w = np.random.rand(L, F) * 2 - 1
	indexes = np.reshape(random.sample(range(0, len(dataset[0]) - 1), L * F), (L, F))
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_weight, b_value, b_score, b_groups = [999] * L, [999] * L, 999, 999, None
	limits = np.zeros((2, F))
	for f in range(F):
		for l in range(L):
			a = np.array(dataset)[:, indexes[l, f]].astype(float) * w[l, f]
			limits[0, f] += np.min(a)
			limits[1, f] += np.max(a)
	for f in range(F):
		for value in np.linspace(limits[0, f], limits[1, f], 10):
			groups = test_split(indexes[:, f], w[:, f], value, dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_weight, b_value, b_score, b_groups = indexes[:, f], w[:, f], value, gini, groups
	return {'index': b_index, 'weight': b_weight, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth,F, L):
    left, right = node['groups']
    #print 'depth = ', depth
    #print len(left), len(right)
    del(node['groups'])
    if not left and not right:
        return
    #check for max depth
    if depth >= max_depth:
        if not left or not right:
            if len(right)!=0:
                node['right'] = node['left'] = to_terminal(right)
            if len(left)!=0:
                node['right'] = node['left'] = to_terminal(left)
        else:
            node['right'],node['left'] = to_terminal(right),to_terminal(left)
        return
    if not left or not right:
        #process right child, left is empty
        if len(right)!=0:
            if len(right)<=min_size:
                node['right']= to_terminal(right)
            else:
                node['right'] = get_split(right,F, L)
                split(node['right'], max_depth, min_size, depth+1,F, L)
            node['left']=node['right']
        if len(left)!=0:
            if len(left)<=min_size:
                node['left']= to_terminal(left)
            else:
                node['left'] = get_split(left,F, L)
                split(node['left'], max_depth, min_size, depth+1,F, L)
            node['right']=node['left']
    else:
        if len(left) <= min_size:
            node['left'] = to_terminal(left)
        else:
            node['left'] = get_split(left,F, L)
            split(node['left'], max_depth, min_size, depth+1,F, L)
        	# process right child
        if len(right) <= min_size:
        		node['right'] = to_terminal(right)
        else:
        		node['right'] = get_split(right,F, L)
        		split(node['right'], max_depth, min_size, depth+1,F, L)


# Build a decision tree
def build_tree(train, max_depth, min_size,F, L = 1):
	# print("Start tree")
	root = get_split(train,F,L)
	split(root, max_depth, min_size, 1,F, L)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if sum([node['weight'][i]*row[node['index'][i]] for i in range(len(node['index']))]) < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)
 
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))
