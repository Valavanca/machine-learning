
config = {
	'Adaboost': {
		 #'pipeline': [('feature_selector', skb),('reduce_dim', pca), ('clf', clf)],
		 'params': dict(feature_selector__k=[4,5,7,8,9,12],
			      reduce_dim__n_components = [1,2,3,4],
			      clf__n_estimators=[100,125,150,175,200,1000,5000],
			      clf__algorithm=['SAMME', 'SAMME.R'])
		},
	'RandomTree': {
		 'params': dict(feature_selector__k=[4,5,6,8,9,11,12],
			      reduce_dim__n_components = [2,3,4],
			      clf__criterion = ['gini','entropy'],
			      clf__min_samples_split = [2,4,6,10])
		},
	'DecisionTree': {
		 'params': dict(feature_selector__k=[5,6,7,8,9,11,12],
			      reduce_dim__n_components = [2,3,4,5],
			      clf__max_features = ['auto', 2],
			      clf__max_depth = [2,4,5,6,8,10],
			      clf__splitter = ['best', 'random']
		            )
		},
	'GaussianNB': {
		 'params': dict(feature_selector__k=[6,7,8,9,11,12],
			      reduce_dim__n_components = [2,3,4,5,6],
		            )
		},
	'MultinomialNB': {
		 'params': dict(feature_selector__k=[5,6,7,8,9,11,12],
			      reduce_dim__n_components = [2,3,4,5],
			      clf__alpha = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
			      #clf__min_samples_split = [2,4,6,10]
		            )
		},
	'SVC': {
		 'params': dict(feature_selector__k=[5,6],
			      reduce_dim__n_components = [2,3],
			      clf__C=[1],
			      clf__kernel=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 
			      #clf__gamma=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
		            )
		},
}
