#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

## USER
from time import time
import tester

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] 
features_list += ['bonus', 'exercised_stock_options', 'total_stock_value'] # POI label
features_list += ['salary', 'deferral_payments', 'total_payments', # financial features 
                  'loan_advances',
                  'deferred_income', 'expenses', 'other', 'long_term_incentive',
                  'restricted_stock', 'director_fees']
features_list += ['to_messages', 'from_poi_to_this_person', # email features
		  'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']  # 'email_address' - string



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)
# remove negative values
for element in data_dict: 
   for feature in features_list[1:]:
       if feature == 'fraction_from_this_person_to_poi' or feature == 'fraction_from_poi_to_this_person': continue
       if data_dict[element][feature] < 0: data_dict[element][feature] = "NaN"


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

##############################################################################
######################           DATA            #############################
 
### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.15, random_state=62)

### Task 4: Try a varity of classifiers 
### Pipe

from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
pca = PCA()

### Feature Scaling: MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)

### Feature selecting: K-Best
skb = SelectKBest(chi2)
skb.fit(features_minmax,labels)

print "\n --- Select K-Bests ---"
for i, s in enumerate((skb.scores_)):
    print " [", s, "] ", features_list[i+1]



###############################################################################
###########################              ######################################
###########################     Tune     ######################################

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
print "\n --- Tune ---"
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

from classifier_param import config
cv = StratifiedShuffleSplit(labels_train, n_iter=8, random_state = 33)

# selection parameters for the algorithm (pipe->grid)
def tune_classifier( algoritm ):
	#t0 = time()
	pipe = Pipeline([('min/max scaler', preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))),
			('feature_selector', algoritm['skb']),
			('reduce_dim', algoritm['pca']), 
			('clf', algoritm['clf'])])
	skl_clf = GridSearchCV(pipe, param_grid=algoritm['params'], cv=cv, scoring='recall', verbose=1)
	#print "\n Searching time: ", round(time()-t0, 3), "s"
	return skl_clf;

# algoritms for loop
algoritms = {
	#'Adaboost': {'skb': skb, 'pca': pca, 'clf': AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy')),
	#		'params': config['Adaboost']['params']
	#		},
	'RandomTree': {'skb': skb, 'pca': pca, 'clf': RandomForestClassifier(),
			'params': config['RandomTree']['params']
			},
	'DecisionTree': {'skb': skb, 'pca': pca, 'clf': DecisionTreeClassifier(criterion='entropy'),
			'params': config['DecisionTree']['params']
			},
	'GaussianNB': {'skb': skb, 'pca': pca, 'clf': GaussianNB(),
			'params': config['GaussianNB']['params']
			},
	#'MultinomialNB': {'skb': skb, 'pca': pca, 'clf': MultinomialNB(),
	#		'params': config['MultinomialNB']['params']
	#		},
	#'SVC': {'skb': skb, 'pca': pca, 'clf': svm.SVC(),
	#		'params': config['SVC']['params']
	#		},
}

# select algoritm from classifier list
temp_score = 0
temp_accuracy = 0
temp_algoritm = ""
clf = MultinomialNB() #default

for item in algoritms:
	t0 = time()
	print "_"*70, "\n|\n|      ", item, "\n|"
	temp_result = tune_classifier(algoritms[item])
	temp_result.fit(features_train,labels_train)
	print "\n Fitting time: ", round(time()-t0, 3), "s"
	
	labels_pred = temp_result.predict(features_test)
	accuracy = accuracy_score(labels_test, labels_pred)
	print " Score: ", temp_result.best_score_
	print " Accuracy: ", accuracy
	print " Best params: ", temp_result.best_params_
	print " _____________"
	clf = temp_result.best_estimator_
	tester.main()
	
	if temp_result.best_score_ > temp_score:
		temp_score = temp_result.best_score_
		temp_accuracy = accuracy
		temp_algoritm = item
		clf = temp_result.best_estimator_
	elif accuracy > temp_accuracy:
		temp_score = temp_result.best_score_
		temp_accuracy = accuracy 
		temp_algoritm = item
		clf = temp_result.best_estimator_

print "\n", "*"*70, "\n* Chosen ", temp_algoritm, ("score: ", temp_score, "accuracy", temp_accuracy),"\n", "*"*70

##############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
print "\n --- Test ---\n", "_"*70

dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

