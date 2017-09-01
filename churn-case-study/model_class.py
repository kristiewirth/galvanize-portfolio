'''
This file contains code to run classification models.
'''

import pandas as pd
import numpy as np
import trio as trio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier

def test_params(model, train_X, train_y, param_list, scoring):
    '''
    Use grid search to discover optimal parameters for each tested model

    INPUT:
    model: fited model
    train_X: training data containing all features
    train_y: training data containing target
    param_list: dictionary of parameters to test and test values
    (e.g., {'alpha': np.logspace(-1, 1, 50)})

    OUTPUT:
    Returns best parameter and its score
    '''
    g = GridSearchCV(model, param_list, scoring=scoring, cv=5, n_jobs=-1, verbose=1)
    g.fit(train_X, train_y)
    print('Model: {}, Best Params: {}, Best Score: {}'\
        .format(model, g.best_params_, g.best_score_))

class Models(object):

    def __init__(self, model_type=LogisticRegression):
        self.model_type = model_type

    def fit(self, train_X, train_y):
        '''
        Fit model on all training data using best parameter value
        '''
        self.model = self.model_type(C = 0.016960624885516043)
        self.model = self.model.fit(train_X, train_y)
        self.train_X = train_X

    def predict(self, test_X):
        '''
        Predict y's for test_X
        '''
        test_y_predicted = self.model.predict(test_X)
        return test_y_predicted

    def score(self, test_y_predicted, test_y, scoring=accuracy_score):
        '''
        Compare predicted y values with y test and compute
        accuracy for entire model
        '''
        score = scoring(test_y, test_y_predicted)
        print('Model: {}\n\nBest Score: {}\n'.format(self.model.__class__.__name__, score))
        return score

    def feature_importance(self):
        coefs = list(self.model.coef_[0])
        features = list(self.train_X.columns)
        arr = []
        for x, y in zip(features, coefs):
            arr.append([x,y])
        arr = np.array(arr)
        arr = arr[np.argsort(arr[:,1])][::-1]
        print('Coefficients:')
        for x in arr:
            print(x)

if __name__ == '__main__':
    pass
    # To achieve consistent results
    # np.random.seed(42)

    # train_X = trio.load_file("../data/churn_train.csv")
    # test_X = trio.load_file("../data/churn_test.csv")
    #
    # train_y = train_X.pop('y')
    # test_y = test_X.pop('y')

    ### FINAL MODELING SECTION ###
    # model = Models(model_type=LogisticRegression)
    # model.fit(train_X, train_y)
    # test_y_predicted = model.predict(test_X)
    # score = model.score(test_y_predicted, test_y, recall_score)
    # model.feature_importance()

    ### LOGISTIC REGRESSION SECTION ###

    # TESTED PARAMS
    # param_list = {'penalty': ['l1', 'l2']}
    # param_list = {'class_weight': ['balanced', None]}
    # param_list = {'class_weight': ['balanced', None]}
    # param_list = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
    # param_list = {'C': np.geomspace(0.09, 0.1, 30), \
    #     'penalty': ['l1', 'l2']}
    # param_list = {'C': [0.016960624885516043]}

    # CODE TO RUN
    # logreg = LogisticRegression()
    # test_params(logreg, train_X, train_y, param_list, 'accuracy')

    # ACCURACY SCORES
    #### Best Params: {'C': 0.016960624885516043}, Best Score: 0.7214

    # Best Params: {'C': 0.098557260196687993, 'solver': 'saga'}, Best Score: 0.721725
    # Best Params: {'C': 0.016450115280080443}, Best Score: 0.721325
    # Best Params: {'solver': 'saga'}, Best Score: 0.72145
    # Best Params: {'C': 0.012742749857031334}, Best Score: 0.72125
    # Best Params: {'penalty': 'l1'}, Best Score: 0.7083421522087503
    # Best Params: {'C': 0.0091029817799152639}, Best Score: 0.7086142775073323
    # Best Params: {'C': 0.0091029817799152639}, Best Score: 0.7086142775073323

    # RECALL SCORES
    # Best Params: {'penalty': 'l1'}, Best Score: 0.8182657289228471
    # Best Params: {'C': 8.7368421052631575}, Best Score: 0.8194979605755229
    # Best Params: {'C': 4.0949150623804189e-20}, Best Score: 1.0

    # PRECISION SCORES
    # Best Params: {'C': 4.094915062380419e-20}, Best Score: 0.5889698545641459

    ### KNN SECTION ###

    # TESTED PARAMS
    # param_list = {'n_neighbors': np.arange(1,11,1)}

    # CODE TO RUN
    # knn = KNeighborsClassifier()
    # test_params(knn, train_X, train_y, param_list, 'accuracy')
    #
    # ACCURACY SCORES
    # Best Params: {'n_neighbors': 9}, Best Score: 0.731725

    ### PROFIT CURVE? ###

    # cost_benefit = cost_benefit()
    # p.profit_curve_main(X_train, X_test, y_train, y_test, cost_benefit)
    # p.sampling_main(LR(), X_train, X_test, y_train, y_test, cost_benefit)

    ### DECISION TREES SECTION ###

    # TESTED PARAMS
    # param_list = {'max_depth': np.arange(5,100,5)}
    # param_list = {'max_depth': np.arange(4,26,2)}
    # param_list = {'max_depth': np.arange(1,12,1)}
    # param_list = {'criterion': ['gini', 'entropy']}
    # param_list = {'splitter': ['best','random']}
    # param_list = {'min_samples_split': np.arange(80,200,20)}
    # param_list = {'max_features': np.arange(1,11,1)}

    # CODE TO RUN
    # dtree = DecisionTreeClassifier()
    # test_params(dtree, train_X, train_y, param_list, 'accuracy')

    # ACCURACY SCORES
    # Best Params: {'max_depth': 10}, Best Score: 0.7571735252320624
    # Best Params: {'max_depth': 8}, Best Score: 0.7606809179693406
    #### Best Params: {'max_depth': 8}, Best Score: 0.7608925709793487
    ### Best Params: {'criterion': 'entropy'}, Best Score: 0.6916820367066792
    ### Best Params: {'splitter': 'random'}, Best Score: 0.6959150969068424
    # Best Params: {'min_samples_split': 92}, Best Score: 0.7553291204305627
    # Best Params: {'min_samples_split': 180}, Best Score: 0.7605902095364799
