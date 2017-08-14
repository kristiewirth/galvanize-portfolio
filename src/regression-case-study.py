'''
This file contains the code used for the regression case study project. This file tests different linears models to predict sale prices from test dataset.
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet

class EqpRegressor(object):
    '''
    Class to fit models
    '''

    def __init__(self, model_type):
        self.model_type = model_type

    def shuffle_data(self, X, y):
        '''
        Shuffle data before analysis

        INPUT:
        X: entire features dataset
        y: target datatset

        OUTPUT:
        Returns shuffled datasets
        '''
        permutation = np.random.permutation(X.shape[0])
        X_s, y_s = X[permutation], y[permutation]
        return X_s, y_s

    def best_params(self, model, X_train, y_train, param_list):
        '''
        Use grid search to discover optimal parameters for each tested model

        INPUT:
        model: fited linear model
        X_train: training data containing all features
        y_train: training data containing target
        param_list: dictionary of parameters to test and test values
            (e.g., {'alpha': np.logspace(-1, 1, 50)})

        OUTPUT:
        Returns best parameter and its negative mean squared error

        '''
        g = GridSearchCV(model, param_list, scoring='neg_mean_squared_error', cv=10)
        g.fit(X_train, y_train)
        return g.best_params_, g.best_score_

    def fit(self, X_train, y_train):
        '''
        Fit model on all training data using best parameter value

        INPUT:
        X_train: training data containing all features
        y_train: training data containing target

        OUTPUT:
        None
        '''
        self.model = self.model_type()
        self.scaler = StandardScaler()
        X_train_standard = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_standard, y_train)

    def predict(self, X_test):
        '''
        Predict y values from X_test data

        INPUT:
        X_test: test data containing all features

        OUTPUT:
        Retuns predicted target values for test data
        '''
        # Only transform test values to standardize; do not fit
        X_test_standard = self.scaler.transform(X_test)
        y_test_predicted = self.model.predict(X_test_standard)
        # Transform negative price predictions to 0
        predictions = []
        for p in y_test_predicted:
            predictions.append(p if p > 0 else 1)
        return y_test_predicted

class Process(object):
    '''
    Class to implement overall case study process
    '''

    def __init__(self, model, train_fn, test_fn, predict_fn):
        self. model = model
        self.train_fn = train_fn
        self.test_fn = test_fn
        self.predict_fn = predict_fn
        self._load_data()
        self._separate_data()

    def _load_data(self):
        '''
        Load data from pickled train & test dataframes.
        '''
        self.train = pd.read_pickle(self.train_fn)
        self.test = pd.read_pickle(self.test_fn)

    def _separate_data(self):
        '''
        Create an extra copy of the data
        '''
        work = self.train.copy()
        self.y_train = work.pop('SalePrice')
        self.X_train = work.values

        work = self.test.copy()
        self.X_test = work.values

    def fit(self):
        '''
        Call fit method
        '''
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        '''
        Call predict method

        OUTPUT:
        Retuns predicted target values for test data
        '''
        predictions = self.model.predict(self.X_test)
        return predictions

    def write_predictions(self, predictions):
        '''
        Write predictions to file

        INPUT:
        predictions: predicted target values for test data

        OUTPUT:
        None
        '''
        sales_predicted = predictions
        sales_id = self.X_test[:,0]
        fh = open(self.predict_fn, 'wt')
        fh.write('SalesID,SalePrice,Usage\n')
        for si, sp in zip(sales_id, sales_predicted):
            fh.write('{:.0f},{:.2f},PublicTest\n'.format(si, sp))
        fh.close()

if __name__ == '__main__':
    # Instantiate chosen model
    model = EqpRegressor(ElasticNet)

    # Run through designed process
    p = Process(model, 'data/train_cleanedpickle_4', 'data/test_cleanedpickle_4', 'data/predict.csv')

    # Fit model
    p.fit()

    # Use model to create predictions for test data
    predictions = p.predict()

    # Write predictions to external file
    p.write_predictions(predictions)
