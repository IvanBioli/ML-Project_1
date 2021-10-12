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

regressors = ['least_squares_GD',
              'least_squares_SGD',
              'least_squares',
              'ridge_regression',
              'logistic_regression',
              'reg_logistic_regression']

params = [{'initial_w': np.ones(tX_train.shape[1]), 'max_iters': 100, 'gamma': 0.1},
          {'initial_w': np.ones(tX_train.shape[1]), 'max_iters': 100, 'gamma': 0.1},
          {},
          {'lambda_': 0.1},
          {'initial_w': np.ones(tX_train.shape[1]), 'max_iters': 100, 'gamma': 0.1},
          {'lambda_': 0.1, 'initial_w': np.ones(tX_train.shape[1]), 'max_iters': 100, 'gamma': 0.1},]

for regressor, param in zip(regressors, params):

    weights, _ = eval(regressor)(y_train, tX_train, **param)
    y_pred = predict_labels(weights, tX_test)
    create_csv_submission(ids_test, y_pred, 'data/submission_' + regressor + '.csv')