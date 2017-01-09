# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 15:31:34 2017

@author: Mathilde
"""


from random import seed
from random import randrange
import numpy as np
import random


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
                #gini += 100
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


# Select the best split point for a dataset
def get_split(dataset, F):
	empty = True
	ind = [0]*(len(dataset[0])-1)
	while empty and sum([i==0 for i in ind])>0:
		indexes = random.sample(range(0,len(dataset[0])-1),F)
		class_values = list(set(row[-1] for row in dataset))
		b_index, b_value, b_score, b_groups = 999, 999, 999, None
		for index in indexes:
			ind[index] += 1
			for row in dataset:
				groups = test_split(index, row[index], dataset)
				gini = gini_index(groups, class_values)
				if gini < b_score:
					b_index, b_value, b_score, b_groups = index, row[index], gini, groups
		# if b_score>0.0:
		# 	empty = ((len(b_groups[0])==0) or (len(b_groups[1])==0))
		# else:
		empty = False
	#print {'index': b_index, 'value': b_value, 'groups': len(b_groups[0]), 'score': b_score}
	return {'index':b_index, 'value':b_value, 'groups':b_groups, 'score':b_score}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth,F):
	left, right = node['groups']
	del(node['groups'])
	# # check for a no split
	# if not left or not right:
	# 	node['left'] = node['right'] = to_terminal(left + right)
	# 	return
	# # check for max depth
	# if depth >= max_depth:
	# 	node['left'], node['right'] = to_terminal(left), to_terminal(right)
	# 	return
	# # process left child
	# if len(left) <= min_size:
	# 	node['left'] = to_terminal(left)
	# else:
	# 	node['left'] = get_split(left,F)
	# 	split(node['left'], max_depth, min_size, depth+1,F)
	# # process right child
	# if len(right) <= min_size:
	# 	node['right'] = to_terminal(right)
	# else:
	# 	node['right'] = get_split(right,F)
	# 	split(node['right'], max_depth, min_size, depth+1,F)
	# check for a no split
	# if not left or not right:
	# 	node['left'] = node['right'] = to_terminal(left + right)
	#  	return
	# #check for max depth
	# if depth >= max_depth:
	# 	node['left'], node['right'] = to_terminal(left), to_terminal(right)
	# 	return
	# #process left child
	if len(left) <= min_size or node['score']<1*10**(-1) or depth >= max_depth:
		if len(left)==0:
			node['left'] = to_terminal([right[0]])
		else:
			node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left,F)
		split(node['left'], max_depth, min_size, depth+1,F)
	# process right child
	if len(right) <= min_size or node['score']<1*10**(-1) or depth >= max_depth:
		if len(right)==0:
			node['right'] = to_terminal([left[0]])
		else:
			node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right,F)
		split(node['right'], max_depth, min_size, depth+1,F)

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