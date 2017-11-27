#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

## USER
from sklearn.feature_selection import f_regression


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] 
features_list += ['bonus', 'exercised_stock_options', 'total_stock_value'] # POI label
features_list += ['salary', 'deferral_payments', 'total_payments', # financial features 
                  'loan_advances', 'restricted_stock_deferred',
                  'deferred_income', 'expenses', 'other', 'long_term_incentive',
                  'restricted_stock', 'director_fees']
features_list += ['to_messages', 'from_poi_to_this_person', # email features
		  'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']  # 'email_address' - string



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for point in my_dataset:
    if 'NaN' in [my_dataset[point]['from_poi_to_this_person'],my_dataset[point]['to_messages']]:
        my_dataset[point]['percent_from_poi_to_this_person'] = 'NaN'
    else:
        my_dataset[point]['percent_from_poi_to_this_person'] = float(my_dataset[point]['from_poi_to_this_person'])/float(my_dataset[point]['to_messages'])

    if 'NaN' in [my_dataset[point]['from_this_person_to_poi'],my_dataset[point]['from_messages']]:
        my_dataset[point]['percent_from_this_person_to_poi'] = 'NaN'
    else:
        my_dataset[point]['percent_from_this_person_to_poi'] = float(my_dataset[point]['from_this_person_to_poi'])/float(my_dataset[point]['from_messages'])

    if 'NaN' in [my_dataset[point]['shared_receipt_with_poi'],my_dataset[point]['to_messages']]:
        my_dataset[point]['percent_shared_receipt'] = 'NaN'
    else:
        my_dataset[point]['percent_shared_receipt'] = float(my_dataset[point]['shared_receipt_with_poi'])/float(my_dataset[point]['to_messages'])

features_list += ['percent_shared_receipt', 'percent_from_poi_to_this_person'] # upd features

#for name in my_dataset:
#	print name, " :: ", my_dataset[name]['from_this_person_to_poi']
#	print name, " :: ", my_dataset[name]['shared_receipt_with_poi']
#	print "______________________________________________________________"


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn.feature_selection import SelectKBest
skb = SelectKBest()
skb.fit(features,labels)

print "\n ********** Select K-Bests **********"
for i, s in enumerate((skb.scores_)):
    print " [", s, "] ", features_list[i+1]


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()

from sklearn.svm import SVC
svc = SVC()

from sklearn.neighbors import NearestCentroid
nc = NearestCentroid()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()


#####  Pipe
print "\n ********** Pipe **********"

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

from sklearn.decomposition import PCA
pca = PCA()

from sklearn.pipeline import Pipeline
estimators = [('scaler', min_max_scaler),('SKB', skb),('reduce_dim', pca), ('clf', nc)]
pipeline = Pipeline(estimators)




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print "\n ********** Tune **********"
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(labels, n_iter=10, random_state = 42)
param_spaces = {
    'SKB__k':[3,4,5,6],
    'reduce_dim__n_components':[2,3],
	# tune    NearestCentroid
    'clf__metric': ['euclidean','manhattan']
}

gs = GridSearchCV(pipeline,param_grid = param_spaces, n_jobs = -1,cv = cv, scoring = 'recall',verbose=10)
gs.fit(features, labels)

print "\n ********** Best **********"
clf = gs.best_estimator_
print "Best params: ", gs.best_params_
# print "Best estimator: ", gs.best_estimator_




# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
print "__________________________________________________"

#dump_classifier_and_data(clf, my_dataset, features_list)

import tester
tester.main()

