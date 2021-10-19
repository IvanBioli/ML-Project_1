# -*- coding: utf-8 -*-
"""
Run
***
Run this file in order to obtain the predictions for each of our regressors.
The parameters are specified in 'params', and the predictions are stored in a
.csv file in directory 'data/submission_[NAME OF THE REGRESSOR]'.

"""

# Package importations
import numpy as np
from proj1_helpers import *
from implementations import *

# Loading and standardizing the training data
y_train, tX_train, _ = load_csv_data('data/train.csv')
tX_train = standardize(tX_train)

# Loading and standardizing the test data
_, tX_test, ids_test = load_csv_data('data/test.csv')

# Final experiment configurations
configs = [{'regressor' : 'least_squares_GD',
            'degrees' : [1, 2, 3],
            'params' : {'initial_w': None, 'max_iters': 1800, 'gamma': 0.009}},

            {'regressor' : 'least_squares_SGD',
            'degrees' : [1, 2, 3],
            'params' : {'initial_w': np.ones(tX_train.shape[1]), 'max_iters': 9800, 'gamma': 0.098}},

           {'regressor' : 'least_squares',
            'degrees' : [1, 2, 3],
            'params' : {}},

           {'regressor' : 'ridge_regression',
            'degrees' : [1, 2, 3],
            'params' : {'lambda_': 1e-3}},

           {'regressor' : 'logistic_regression',
            'degrees' : [1, 2, 3],
            'params' : {'initial_w': np.ones(tX_train.shape[1]), 'max_iters': 100, 'gamma': 0.1}},

           {'regressor' : 'reg_logistic_regression',
            'degrees' : [1, 2, 3],
            'params' : {'lambda_': 0.1, 'initial_w': np.ones(tX_train.shape[1]), 'max_iters': 100, 'gamma': 0.1}}]


for config in configs:

    tX_train_poly = polynomial_basis(tX_train, config['degrees'], std=True)
    tX_test_poly = polynomial_basis(tX_test, config['degrees'], std=True)

    weights, _ = eval(config['regressor'])(y_train, tX_train_poly, **config['params'])
    y_pred = predict_labels(weights, tX_test_poly)
    create_csv_submission(ids_test, y_pred, 'data/submission_' + config['regressor'] + '.csv')