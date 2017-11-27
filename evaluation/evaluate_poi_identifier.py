#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import tree
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn import cross_validation

## Split data
featTrain, featTest, labelTrain, labelTest = cross_validation.train_test_split(features, labels, test_size = 0.3, random_state = 42)
print "Sum ", sum(labelTest)
print "People ", len(labelTest)

## Fit tree
clf = tree.DecisionTreeClassifier()
clf.fit(featTrain, labelTrain)

## Predict
predict = clf.predict(featTest)
#predict = [1.] * 29

print "precision score: ", precision_score(labelTest, predict )
print "recall score: ", recall_score(labelTest, predict )

print "accuracy: ", accuracy_score(labelTest, predict)

print " LOL ", precision_score([0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] , [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])



