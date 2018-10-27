#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

### USER
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.ensemble import (AdaBoostClassifier,BaggingClassifier, RandomForestClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=10)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "AdaBoost+tree_accuracy"
print accuracy_score(labels_test, pred)


clf = AdaBoostClassifier(base_estimator=SVC(random_state=1), algorithm="SAMME", n_estimators=1)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "svc_accuracy"
print accuracy_score(labels_test, pred)

clf_randomforest = RandomForestClassifier(n_estimators=100)
clf_randomforest.fit(features_train, labels_train)
prettyPicture(clf_randomforest, features_test, labels_test)
#score.append(["randomforest", clf_randomforest.score(features_test, labels_test)])
pred = clf_randomforest.predict(features_test)
print "randomforest_accuracy"
print accuracy_score(labels_test, pred)




try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
