import pandas
import numpy as np
from numpy import repeat as rep
from numpy import concatenate as cat
import random 

DX = 0.001
MIN_GROUP_SIZE = 5 # >= 4
HUGE_NUMBER = 99999
TOL = 0.001

class Node:
  def __init__(self, IDS, depth):
    self.IDS, self.depth = IDS, depth
    self.pred = sum([data[i][-1] for i in IDS])/len(IDS)
    self.SS = sum([(data[i][-1]-self.pred)**2 for i in self.IDS])/len(self.IDS)
    self.lc, self.rc, self.sf, self.sv = None, None, None, None
 
  #def isPure(self): return all([data[i][-1]==data[self.IDS[0]][-1] for i in self.IDS])
 
  def updateNode(self, lc, rc, sf, sv):
    self.lc, self.rc, self.sf, self.sv = lc, rc, sf, sv

  def split_node(self):
    split = self.find_split()

    if self.SS-self.SS_parts(split['group1'], split['group2']) > TOL:
      lc, rc = Node(split['group1'], self.depth+1), Node(split['group2'], self.depth+1)
      self.updateNode(lc, rc, split['feature'], split['value'])
      if len(lc.IDS) >= MIN_GROUP_SIZE: self.lc.split_node()
      if len(rc.IDS) >= MIN_GROUP_SIZE: self.rc.split_node()

  # compute squared sum of residuals given groups
  def SS_parts(self, IDS1, IDS2):
    y1Hat = sum([data[i][-1] for i in IDS1])/len(IDS1)
    y2Hat = sum([data[i][-1] for i in IDS2])/len(IDS2)
    return (sum([(data[i][-1]-y1Hat)**2 for i in IDS1]) + sum([(data[i][-1]-y2Hat)**2 for i in IDS2]))/(len(IDS1)+len(IDS2))

  # left side contain <= in numerical and == in nominal case
  def split_at(self, feature, value):
    func = None
    if isinstance(value, (int, float)):
      func = lambda ID: data[ID][feature] <= value
    else:
      func = lambda ID: data[ID][feature] == value # equality forms one partition
    p1 = [ID for ID in self.IDS if func(ID)]
    p2 = [ID for ID in self.IDS if not func(ID)]
    return p1, p2

  def find_split(self, RF = False):
    optFeat, optVal, optCost, optP1, optP2 = None, None, HUGE_NUMBER, None, None
    for feature in range(len(data[0])-1):
      splitPoint = None
      if isinstance(data[0][feature], (int, float)):
        for i in self.IDS:
          splitPoint = data[i][feature]+DX
          p1, p2 = self.split_at(feature, splitPoint)
          if len(p1) > 0 and len(p2) > 0:
            val = self.SS_parts(p1, p2)
            if val < optCost:
              optFeat, optVal, optCost, optP1, optP2 = feature, splitPoint, val, p1, p2
      else:
        for i in self.IDS:
          splitPoint = data[i][feature]
          p1, p2 = self.split_at(feature, splitPoint)
          if len(p1) > 0 and len(p2) > 0:
            val = self.SS_parts(p1, p2)
            if val < optCost:
              optFeat, optVal, optCost, optP1, optP2 = feature, splitPoint, val, p1, p2
    return {'feature': optFeat, 'value': optVal, 'group1': optP1, 'group2': optP2, 'optCost': optCost}
 
  def represent_tree(self, child = ''):
    if self.depth==1: print('1: ', end='')
    print('%d observations, mean: %.4f\n' % (len(self.IDS), self.pred), end='')
    if self.sf is not None:
      if isinstance(self.sv, (int, float)):
        # leftchild
        print(child+'  %d: X[%d]<=%.4f  --  ' % (self.depth+1, self.sf, float(self.sv)), end='')
        self.lc.represent_tree(child+'  ')

        # rightchild
        print(child+'  %d: X[%d]>%.4f  --  ' % (self.depth+1, self.sf, float(self.sv)), end='')
        self.rc.represent_tree(child+'  ')
 
      else:
        # leftchild
        print(child+'  %d: X[%d]=%s  --  ' % (self.depth+1, self.sf, self.sv), end='')
        self.lc.represent_tree(child+'  ')

        # rightchild
        print(child+'  %d: X[%d] != %s  --  ' % (self.depth+1, self.sf, self.sv), end='')
        self.rc.represent_tree(child+'  ')
  
  def predict(self, X):
    if self.sf is None: return self.pred
    else:
      if isinstance(self.sv, (int, float)):
        if X[self.sf] <= self.sv: return self.lc.predict(X)
        else: return self.rc.predict(X)
      else:
        if X[self.sf] == self.sv: return self.lc.predict(X)
        else: return self.rc.predict(X)

###################################################################################################
# response is last column
#data = [[1, 1, 2], [2, 2, 2], [3, 1, 3], [1, 7, 5], [2, 8, 5], [3, 6, 6]]
#data = [['A','Y', 3],['A', 'X', 7], ['C','Y',3], ['B','X',8], ['B','Y',4], ['C','X',6]]
#dataORG = pandas.read_csv('servo', header=None).values.tolist()
#dataORG = pandas.read_csv('abalone', header=None).values.tolist()
# THe author has only 12 columns, I suppose he skipped the 4th
dataORG = pandas.read_csv('housing', header=None, sep=r"\s+", usecols=[0,1,2,4,5,6,7,8,9,10,11,12,13]).values.tolist()

#for row in data[1:5]: print(row)

#data = dataOrg
#root = Node(range(len(data)), 1)
#root.split_node()
#root.represent_tree()

# Bootstrap Agregation - 90% training, 10% test.
# fraction first, 1- fraction second
def bootstrap_OOB(fraction):
  n = round(len(dataORG) * fraction)
  bs = [random.randrange(len(dataORG)) for i in range(n)]
  OOB = [elm for elm in range(len(dataORG)) if elm not in bs]
  return bs, OOB

'''
reps = 100; errorsBagging = [0]*100
i=0#for i in range(100):
bs, OOB = bootstrap_OOB(0.9)
data = [dataORG[r] for r in bs]
root = Node(range(len(data)), 1)
root.split_node()
errorsBagging[i] = sum([(root.predict(dataORG[i][:-1])-dataORG[i][-1])**2 for i in OOB])/len(OOB)
# random forest
numFet = 5; # each uniform linear combination
features = [[0 for k in range(len(data[0]-1))] for l in range(numFet)]
for j in range(numFet):
  [a,b] = random.sample(range(len(data[0]-1)), 2)
  features[j][a], features[j][b] = random.uniform(-1,1), random.uniform(-1,1)
  
  print(features[j])
#print(features)   
'''




''' proceedures is moved into classfunctions.
# compute squared sum of residuals given predictions
def SS(yHat, Y): return np.dot(Y-yHat, Y-yHat)

# compute squared sum of residuals given groups
def SS_parts(IDS1, IDS2):
  y1Hat = sum([data[i][-1] for i in IDS1])/len(IDS1)
  y2Hat = sum([data[i][-1] for i in IDS2])/len(IDS2)
  return (sum([(data[i][-1]-y1Hat)**2 for i in IDS1]) + sum([(data[i][-1]-y2Hat)**2 for i in IDS2]))/(len(IDS1)+len(IDS2))

# left side contain <= in numerical and == in nominal case
def split_at(feature, value, IDS):
  func = None
  if isinstance(value, (int, float)):
    func = lambda ID: data[ID][feature] <= value
  else:
    func = lambda ID: data[ID][feature] == value # equality forms one partition
  p1 = [ID for ID in IDS if func(ID)]
  p2 = [ID for ID in IDS if not func(ID)]
  return p1, p2

def find_split(IDS):
  optFeat, optVal, optCost, optP1, optP2 = None, None, HUGE_NUMBER, None, None
  for feature in range(len(data[0])-1):
    splitPoint = None
    if isinstance(data[0][feature], (int, float)):
      for i in IDS:
        splitPoint = data[i][feature]+DX
        p1, p2 = split_at(feature, splitPoint, IDS)
        if len(p1) > 0 and len(p2) > 0:
          val = SS_parts(p1, p2)
          if val < optCost:
            optFeat, optVal, optCost, optP1, optP2 = feature, splitPoint, val, p1, p2
    else:
      for i in IDS:
        splitPoint = data[i][feature]
        p1, p2 = split_at(feature, splitPoint, IDS)
        if len(p1) > 0 and len(p2) > 0:
          val = SS_parts(p1, p2)
          if val < optCost:
            optFeat, optVal, optCost, optP1, optP2 = feature, splitPoint, val, p1, p2
  return {'feature': optFeat, 'value': optVal, 'group1': optP1, 'group2': optP2, 'optCost': optCost}
'''
