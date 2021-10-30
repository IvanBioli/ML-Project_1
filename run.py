#%% -*- coding: utf-8 -*-
"""
Run
***
Run this file in order to obtain the predictions for each of our regressors.
The parameters are specified in 'params', and the predictions are stored in a
.csv file in directory 'data/submission_[NAME OF THE REGRESSOR]'.
"""

# Package importations
from proj1_helpers import *
from implementations import *

# Loading the training data
y_train, tX_train, _ = load_csv_data('data/train.csv')

# Loading the test data
_, tX_test, ids_test = load_csv_data('data/test.csv')

# Determine the ratio of samples with label -1 to the total samples
pred_ratio = sum(y_train == -1) / len(y_train)

# Outlier filtering procedures (only uses properties of the training set)
tX_train_subs_999 = substitute_999(tX_train, tX_train, 'median')
tX_test_subs_999 = substitute_999(tX_train, tX_test, 'median')
tX_train_subs_outliers = substitute_outliers(tX_train_subs_999, tX_train_subs_999, 'mean', 3)
tX_test_subs_outliers = substitute_outliers(tX_train_subs_999, tX_test_subs_999, 'mean', 3)

configs = [{'regressor' : 'least_squares_GD',
            'degrees' : [1, 2, 3],
            'params' : {'gamma': 9e-2, 'max_iters': 200},
            'pred_ratio' : pred_ratio,
            'tX_train' : tX_train_subs_outliers,
            'tX_test' : tX_test_subs_outliers},

            {'regressor' : 'least_squares_SGD',
            'degrees' : [1, 2],
            'params' : {'gamma': 3e-4, 'max_iters': 10000, 'seed': 0},
            'pred_ratio' : pred_ratio,
            'tX_train' : tX_train_subs_outliers,
            'tX_test' : tX_test_subs_outliers},

           {'regressor' : 'least_squares',
            'degrees' : [1, 2, 3, 4],
            'params' : {},
            'pred_ratio' : pred_ratio,
            'tX_train' : tX_train_subs_outliers,
            'tX_test' : tX_test_subs_outliers},

           {'regressor' : 'ridge_regression',
            'degrees' : [1, 2, 3, 4, 5, 6, 7],
            'params' : {'lambda_': 1e-8},
            'pred_ratio' : pred_ratio,
            'tX_train' : tX_train_subs_outliers,
            'tX_test' : tX_test_subs_outliers},

            {'regressor' : 'logistic_regression',
            'degrees' : [0, 1, 2, 3, 4],
            'params' : {'gamma': 2e-6, 'max_iters': 500},
            'pred_ratio' : False,
            'tX_train' : tX_train_subs_outliers,
            'tX_test' : tX_test_subs_outliers},

           {'regressor' : 'reg_logistic_regression',
            'degrees' : [0, 1, 2, 3, 4],
            'params' : {'lambda_': 1e-4, 'gamma': 2.5e-6, 'max_iters': 500},
            'pred_ratio' : False,
            'tX_train' : tX_train_subs_outliers,
            'tX_test' : tX_test_subs_outliers}]

for config in configs:

    # Raise the sets to a polynomial basis (and standardize them simultaneously)
    tX_train_poly = polynomial_basis(config['tX_train'], config['degrees'], True, config['tX_train'])
    tX_test_poly = polynomial_basis(config['tX_train'], config['degrees'], True, config['tX_test'])

    # Fitting the regressor configurations and creating predictions
    weights, _ = eval(config['regressor'])(y_train, tX_train_poly, **config['params'])
    y_pred = predict_labels(weights, tX_test_poly, config['pred_ratio'])
    create_csv_submission(ids_test, y_pred, 'data/submission_' + config['regressor'] + '.csv')

# %%
