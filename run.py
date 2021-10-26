# -*- coding: utf-8 -*-
"""
Run
***
Run this file in order to obtain the predictions for each of our regressors.
The parameters are specified in 'params', and the predictions are stored in a
.csv file in directory 'data/submission_[NAME OF THE REGRESSOR]'.

Minimum working example
-----------------------
>>> from proj1_helpers import *
>>> from implementations import *
>>> y_train, tX_train, _ = load_csv_data('data/train.csv')
>>> tX_train = standardize(tX_train)
>>> print(least_squares_GD(y_train, tX_train))
>>> print(least_squares_SGD(y_train, tX_train))
>>> print(least_squares(y_train, tX_train))
>>> print(ridge_regression(y_train, tX_train))
>>> print(logistic_regression(y_train, tX_train))
>>> print(reg_logistic_regression(y_train, tX_train))
"""

# Package importations
from proj1_helpers import *
from implementations import *

# Loading the training data
y_train, tX_train, _ = load_csv_data('data/train.csv')

# Loading the test data
_, tX_test, ids_test = load_csv_data('data/test.csv')

# Final experiment configurations
configs = [{'regressor' : 'least_squares_GD',
            'degrees' : [1, 2],
            'params' : {'gamma': 0.1}},

            {'regressor' : 'least_squares_SGD',
            'degrees' : [1, 2, 3],
            'params' : {'gamma': 0.0001}},

           {'regressor' : 'least_squares',
            'degrees' : [1, 2, 3, 4, 5],
            'params' : {}},

           {'regressor' : 'ridge_regression',
            'degrees' : [1, 2, 3, 4, 5],
            'params' : {'lambda_': 1e-5}},

           {'regressor' : 'logistic_regression',
            'degrees' : [0, 1, 2, 3],
            'params' : {'gamma': 0.001}},

           {'regressor' : 'reg_logistic_regression',
            'degrees' : [0, 1],
            'params' : {'lambda_': 0.0001}}]


for config in configs:

    # Raise the sets to a polynomial basis (and standardize them simultaneously)
    tX_train_poly = polynomial_basis(tX_train, config['degrees'], std=True)
    tX_test_poly = polynomial_basis(tX_test, config['degrees'], std=True)

    # Fitting the regressor configurations and creating predictions
    weights, _ = eval(config['regressor'])(y_train, tX_train_poly, **config['params'])
    y_pred = predict_labels(weights, tX_test_poly)
    create_csv_submission(ids_test, y_pred, 'data/submission_' + config['regressor'] + '.csv')
