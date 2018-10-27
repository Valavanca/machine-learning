#!/usr/bin/python

import sys
import os
import pickle
import numpy as np

if __file__=="poi_id.py":
	sys.path.append("../tools/")
else:
	sys.path.append(os.getcwd())
	sys.path.append("./tools/")
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

## USER
from time import time
import tester
from sklearn.metrics import recall_score

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
with open(os.path.join(__location__, "final_project_dataset.pkl"), "r") as data_file:
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

# upd 27.03
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# import seaborn as sns

from classifier_param import config
cv = StratifiedShuffleSplit(labels_train, n_iter=8, random_state = 33)

# selection parameters for the algorithm (pipe->grid)
def tune_classifier( algoritm ):
	pipe = Pipeline([('min/max scaler', preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))),
			('feature_selector', algoritm['skb']),
			('reduce_dim', algoritm['pca']),
			('clf', algoritm['clf'])])
	#return GridSearchCV(pipe, param_grid=algoritm['params'], cv=cv, scoring='recall', verbose=10)			
	return GridSearchCV(pipe, param_grid=algoritm['params'], scoring='accuracy', verbose=1)


# algoritms for loop
algoritms = {
	'Adaboost': {'skb': skb, 'pca': pca, 'clf': AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy')),
			'params': config['Adaboost']['params']
			},
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
def arr_scaler(arr):
	temp_max = max(arr)
	return map(lambda x: x/temp_max, arr)

# PLOT function
def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2, alg_name):
	# Get Test Scores Mean, time and std for each grid search
	scores_mean = cv_results['mean_test_score']
	scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

	scores_sd = cv_results['std_test_score']
	scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

	fit_time = cv_results['mean_fit_time']
	fit_time = arr_scaler(fit_time)
	fit_time = np.array(fit_time).reshape(len(grid_param_2),len(grid_param_1))

	# Plot Grid search scores
	_, ax = plt.subplots(1,1)
	color = ["#1f77b4", "#2ca02c", "#d62728", "#d62728", "#9467bd"]

	for idx, val in enumerate(grid_param_2):
		print "param_2", idx, val
		ax.plot(grid_param_1, scores_mean[idx,:], '-o', color=color[idx%len(color)], label= name_param_2 + ': ' + str(val))
		ax.plot(grid_param_1, fit_time[idx,:], '-x', linestyle=':', color=color[idx%len(color)])

	ax.set_title(alg_name, fontsize=20, fontweight='bold')
	ax.set_xlabel(name_param_1, fontsize=16)
	ax.set_ylabel('Accuracy / Time', fontsize=16)
	ax.legend(loc="best", fontsize=15)
	ax.grid('on')

# select algoritm from classifier list
temp_recall = 0
temp_score = 0
temp_accuracy = 0
temp_algoritm = ""
clf = MultinomialNB() #default

for item in algoritms:
	# t0 = time()
	# Algorithm name
	print "_"*70, "\n|\n|      ", item, "\n|"

	temp_result = tune_classifier(algoritms[item])
	temp_result.fit(features_train,labels_train)

	# -----------------------------------------
	# PLOT
	print "Grid scores on development set:\n"
	means = temp_result.cv_results_['mean_test_score']
	stds = temp_result.cv_results_['std_test_score']
	fit_time = temp_result.cv_results_['mean_fit_time']
	for mean, std, time, params in zip(means, stds, fit_time, temp_result.cv_results_['params']):
		print("%0.3f (+/-%0.03f, %0.2f s*10^5) for %r"
			% (mean, std * 2, time*100000, params))
	print("\n")

	plot_grid_search(temp_result.cv_results_, range(4,19), [2,3,4],'Features','reduce', item)
	plt.show()


	# REPORT
	print("Detailed classification report:\n")
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.\n")

	labels_pred = temp_result.predict(features_test)
	pred = temp_result.predict(features)

	print(classification_report(labels_test, labels_pred))
	
	
	accuracy = accuracy_score(labels_test, labels_pred)
	print " Score: ", temp_result.best_score_
	print " Accuracy: ", accuracy
	print " Best params: ", temp_result.best_params_
	print " _____________"

	### launch main test for all the estimator cases
	#dump_classifier_and_data(temp_result.best_estimator_, my_dataset, features_list)
	#tester.main()
		
	focus_recall = recall_score(labels, pred, average='macro')
	print " Recall: ", focus_recall
	if focus_recall > temp_recall:
		temp_recall = focus_recall
		temp_score = temp_result.best_score_
		temp_accuracy = accuracy
		temp_algoritm = item
		clf = temp_result.best_estimator_
	elif accuracy > temp_accuracy:
		temp_recall = focus_recall
		temp_score = temp_result.best_score_
		temp_accuracy = accuracy 
		temp_algoritm = item
		clf = temp_result.best_estimator_
	

print "\n", "*"*70, \
	"\n* Chosen ", temp_algoritm, ("recall: ", temp_recall, "accuracy", temp_accuracy),"\n", \
	"*"*70

##############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
print "\n --- Test ---\n", "_"*70

dump_classifier_and_data(clf, my_dataset, features_list)

if __name__ == '__main__':
	tester.main()






