#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
print "[start geting email"
features_train, features_test, labels_train, labels_test = preprocess()
print "..emails have got]"



#########################################################
from tpot import TPOTClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

print len(features_train[0])

clf = tree.DecisionTreeClassifier(min_samples_split=40)
print "[1] Fit.."
t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time: ", round(time()-t0, 3), "s"

print "[2] Pred.."
t1 = time()
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "predicting time: ", round(time()-t1, 3), "s"

print "Accuracy :: "
print(round(acc,3))

print "______________\n TPOT :: \n______________"
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(features_train, labels_train)
print(tpot.score(features_test, labels_test))


#########################################################


