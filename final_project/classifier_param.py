
config = {
	'Adaboost': {
		 #'pipeline': [('feature_selector', skb),('reduce_dim', pca), ('clf', clf)],
		 'params': dict(feature_selector__k=[4,5,6,8],
			      reduce_dim__n_components = [1,2,3,4],
			      clf__n_estimators=[100,125,150,175,200,1000],
			      clf__algorithm=['SAMME', 'SAMME.R'])
		},
	'RandomTree': {
		 'params': dict(feature_selector__k=[4,5,6,7,8,9],
			      reduce_dim__n_components = [2,3,4],
			      clf__criterion = ['gini','entropy'],
			      clf__min_samples_split = [2,4,6,10])
		},
	'DecisionTree': {
		 'params': dict(feature_selector__k=[5,6,7,8,9],
			      reduce_dim__n_components = [2,3,4,5],
			      clf__max_features = ['auto', 2],
			      #clf__min_samples_split = [2,4,6,10]
		            )
		},
}
