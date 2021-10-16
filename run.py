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
tX_test = standardize(tX_test)

# Derived data sets
tX_test_p123 = polynomial_basis(tX_test, [1, 2, 3], std=True)
tX_train_p123 = polynomial_basis(tX_train, [1, 2, 3], std=True)
tX_test_p123456 = polynomial_basis(tX_test, [1, 2, 3, 4, 5, 6], std=True)
tX_train_p123456 = polynomial_basis(tX_train, [1, 2, 3, 4, 5, 6], std=True)


# Experiment configuration
regressors = ['least_squares_GD',
              'least_squares',
              'ridge_regression']

sets = [{'tX_train' : tX_train_p123, 'tX_test' : tX_test_p123},
        {'tX_train' : tX_train_p123456, 'tX_test' : tX_test_p123456},
        {'tX_train' : tX_train_p123456, 'tX_test' : tX_test_p123456}]

params = [{'initial_w': np.ones(tX_train_p123.shape[1]), 'max_iters': 2000, 'gamma': 0.04},
          {},
          {'lambda_': 1e-1}]

"""
# Final experiment configurations
regressors = ['least_squares_GD',
              'least_squares_SGD',
              'least_squares',
              'ridge_regression',
              'logistic_regression',
              'reg_logistic_regression']

sets = [{'tX_train' : tX_train_p123456, 'tX_test' : tX_test_p123456},
        {'tX_train' : tX_train, 'tX_test' : tX_test},
        {'tX_train' : tX_train, 'tX_test' : tX_test},
        {'tX_train' : tX_train_p123456, 'tX_test' : tX_test_p123456},
        {'tX_train' : tX_train, 'tX_test' : tX_test},
        {'tX_train' : tX_train, 'tX_test' : tX_test}]

params = [{'initial_w': np.ones(tX_train.shape[1]), 'max_iters': 1800, 'gamma': 0.009},
          {'initial_w': np.ones(tX_train.shape[1]), 'max_iters': 9800, 'gamma': 0.098},
          {},
          {'lambda_': 2e-5},
          {'initial_w': np.ones(tX_train.shape[1]), 'max_iters': 100, 'gamma': 0.1},
          {'lambda_': 0.1, 'initial_w': np.ones(tX_train.shape[1]), 'max_iters': 100, 'gamma': 0.1},]
"""


for regressor, set, param in zip(regressors, sets, params):

    weights, _ = eval(regressor)(y_train, set['tX_train'], **param)
    y_pred = predict_labels(weights, set['tX_test'])
    create_csv_submission(ids_test, y_pred, 'data/submission_' + regressor + '.csv')