# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 15:31:34 2017

@author: Mathilde
"""

# CART on the Bank Note dataset
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
#
## Calculate accuracy percentage
#def accuracy_metric(actual, predicted):
#	correct = 0
#	for i in range(len(actual)):
#		if actual[i] == predicted[i]:
#			correct += 1
#	return correct / float(len(actual)) * 100.0
#
## Evaluate an algorithm using a cross validation split
#def evaluate_algorithm(dataset, algorithm, n_folds, *args):
#	folds = cross_validation_split(dataset, n_folds)
#	scores = list()
#	for fold in folds:
#		train_set = list(folds)
#		train_set.remove(fold)
#		train_set = sum(train_set, [])
#		test_set = list()
#		for row in fold:
#			row_copy = list(row)
#			test_set.append(row_copy)
#			row_copy[-1] = None
#		predicted = algorithm(train_set, test_set, *args)
#		actual = [row[-1] for row in fold]
#		accuracy = accuracy_metric(actual, predicted)
#		scores.append(accuracy)
#	return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
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
def get_split(dataset, F):
    indexes = random.sample(range(0,len(dataset[0])-1),F)
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None 
    for index in indexes:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    #print 'b_index = ', b_index
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth,F):
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
                node['right'] = get_split(right,F)
                split(node['right'], max_depth, min_size, depth+1,F)
            node['left']=node['right']
        if len(left)!=0:
            if len(left)<=min_size:
                node['left']= to_terminal(left)
            else:
                node['left'] = get_split(left,F)
                split(node['left'], max_depth, min_size, depth+1,F)
            node['right']=node['left']
    else:
        if len(left) <= min_size:
            node['left'] = to_terminal(left)
        else:
            node['left'] = get_split(left,F)
            split(node['left'], max_depth, min_size, depth+1,F)
        	# process right child
        if len(right) <= min_size:
        		node['right'] = to_terminal(right)
        else:
        		node['right'] = get_split(right,F)
        		split(node['right'], max_depth, min_size, depth+1,F)
            
#	# check for a no split
#	if not left or not right:
#		node['left'] = node['right'] = to_terminal(left + right)
#		return
	# check for max depth
#	if depth >= max_depth:
#		node['left'], node['right'] = to_terminal(left), to_terminal(right)
#		return
	

# Build a decision tree
def build_tree(train, max_depth, min_size,F):
	root = get_split(train,F)
	split(root, max_depth, min_size, 1,F)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
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
 

# Test CART on Bank Note dataset
#seed(1)
# load and prepare data
#filename = 'sonar.csv'
#dataset = load_csv(filename)
## convert string attributes to integers
#for i in range(len(dataset[0])):
#	str_column_to_float(dataset, i)
## evaluate algorithm
#n_folds = 5
#max_depth = 5
#min_size = 10
#scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
#print('Scores: %s' % scores)
#print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))