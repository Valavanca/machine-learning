## Intro to Machine Learning :chart_with_upwards_trend:

It is completed [Intro to Machine Learning course](https://classroom.udacity.com/courses/ud120) with a final project. In the end of completed course building a program to identify Enron Employees who may have committed fraud based on the public Enron dataset.

- - - -

## Installation

### Dependences:
* scikit-learn
* nltk

### Run
`python final_project/poi_id.py`

Entry point for final project is in the _/final_project/poi_id.py_
This code prepares data and chose the classifier with the best parameters. Then call _/final_project/tester.py_ for generates the necessary .pkl files for validating results.
>Execute tester.py it is the faster way to check prepared algorithm from poi_id.py

## Questions



***1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those? [relevant rubric items: “data exploration”, “outlier investigation”] 

The goal of this project is identified fraud in a person's dataset. We can manipulate with information of resealing companies every month, personal emails, selling assets, getting a bonus and similar staff. If we will know a financial activity from another person, hopefully, we can detect an anomaly and the basis of the crime based on the financial features and emails, whether the person is actually POI.

There are 146 people in the dataset and 18 of those are a person of interest (actually 30+). There are 21 features in the dataset.

In the dataset, there are outliers which are 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK'. This numerical features that every person in ENRON dataset has but counted as a person. We should pop out this because it's not a relevant information.


*** 2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importance of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values. [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”] </strong>

I added new features such as fraction in which this person sent email to POI persons, and fraction POI persons send emails to this persons. This values can have to reveal relationships between POI. But they have not a huge effect on the performance.
For feature selecting I used `SelectKBest` and `PCA` (Principal component analysis). I scaled numerical features because I don't know how much features I'll have. In this case, if one of the features has a broad range of values, the distance will be governed by this particular feature. 


*** 3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]</strong>

In the ended, I chose the Adaboost with DecisionTree. Additionally, I tried to estimate simple `DecisionTree`, `RandomTree`, `GaussianNB`. All classifier tuned using `GridSearchCV`

```python
### DecisionTree
Best params:  {'clf__splitter': 'best', 'feature_selector__k': 6, 'clf__max_features': 2, 'clf__max_depth': 8, 'reduce_dim__n_components': 3}
###  Score: 0.5  Accuracy: 0.727272727273 Recall: 0.876888888889
```
```python
### RandomTree
Best params:  {'feature_selector__k': 12, 'clf__criterion': 'gini', 'reduce_dim__n_components': 2, 'clf__min_samples_split': 6}
###  Score: 0.3125 Accuracy: 0.818181818182 Recall: 0.761777777778
```
```python
### GaussianNB
Best params:  {'feature_selector__k': 8, 'reduce_dim__n_components': 6}
###  Score: 0.375 Accuracy: 0.772727272727 Recall: 0.658444444444
```

```python
### Adaboost [Best] 
Best params:  {'feature_selector__k': 12, 'clf__algorithm': 'SAMME', 'reduce_dim__n_components': 3, 'clf__n_estimators': 1000}
###  Score: 0.4375 Accuracy: 0.818181818182 Recall: 0.888888888889
```
Сheck Adaboost in the main tester: :checkered_flag:
```xml
	Accuracy: 0.83753	Precision: 0.38959	Recall: 0.38550	
    F1: 0.38753	F2: 0.38631
```


*** 4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier). [relevant rubric item: “tune the algorithm”]</strong>


Without tuning the parameters probably miss the best combination because it can not be predicted. For selecting algorithms with the best parameters I compare recall score of prediction.
For every one case with the algorithm, I tuned count of features using (`SKB`), principal component analysis(`PCA`) and `StratifiedShuffleSplit` with them. Additionally, I took a range of potentially good special parameters of the algorithm, if they were. The scoring method used is a Recall score.


*** 5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]</strong>

Validation is important when we want to test the model against future data. We can't train the model using whole data and test it with the same one. I used `StratifiedShuffleSplit` for more smooth result in `GridSearchCV` and save a ~15% of data to see the general situation in compares algorithms. 


*** 6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]</strong>

I will use recall and accuracy for my evaluation metrics. I stopped at this choice because other meters sometimes gave the same results. From the performance that I got, I have good precision and good recall. That means the model is able to identify the when the real POI comes out, and have the good probability of flagging POI person. 


## Reference

* [Intro to Machine Learning](https://classroom.udacity.com/courses/ud120)
* [POI from life](http://www.nytimes.com/packages/html/national/20061023_ENRON_TABLE/index.html)

