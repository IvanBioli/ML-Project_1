# -*- coding: utf-8 -*-
"""
Run
***
Run this file in order to obtain the predictions for our optimized regressor.
Uncomment the codeblock further down, to reproduce all results we obtained.
The predictions are compared to the ones we submitted to AIcrowd, and are then
stored in a .csv file in directory 'data/submission_[NAME OF THE REGRESSOR]'.

Time to execute: ~1 Min (~4 Min for whole reproduction)
Required RAM: 6 GB
"""

# Package importations
from proj1_helpers import *
from implementations import *

# Loading the training data
y_train, tX_train, _ = load_csv_data('data/train.csv')

# Loading the test data
_, tX_test, ids_test = load_csv_data('data/test.csv')

config = {'regressor' : 'optimized_regression',
          'base_regressor' : 'ridge_regression',
          'degrees' : 9,
          'params' : {'lambda_': 1e-13},
          'tX_train' : tX_train,
          'tX_test' : tX_test}

y_pred = optimized_regression(config['base_regressor'], config['tX_train'], y_train, config['tX_test'], config['degrees'], config['params'])
create_csv_submission(ids_test, y_pred, 'data/submission_' + config['regressor'] + '_' + config['base_regressor'] + '.csv')
y_pred_submission = np.genfromtxt('data/final_submission_' + config['regressor'] + '_' + config['base_regressor'] + '.csv', delimiter=",", skip_header=1, dtype=int, usecols=1)
print(config['regressor'], 'with', config['base_regressor'], 'found', sum(y_pred != y_pred_submission), 'different predictions from submissions.')

"""
# Uncomment to reproduce all results stated in Table 2 of the report

# Outlier filtering procedures (by only using properties of the training set)
tX_test_subs = substitute_999(tX_train, tX_test, 'median')
tX_train_subs = substitute_999(tX_train, tX_train, 'median')
tX_test_subs = substitute_outliers(tX_train_subs, tX_test_subs, 'mean', 3)
tX_train_subs = substitute_outliers(tX_train_subs, tX_train_subs, 'mean', 3)

# Additional regressor configurations
configs = [{'regressor' : 'least_squares_GD',
            'degrees' : 4,
            'params' : {'gamma': 7e-2, 'max_iters': 200},
            'tX_train' : tX_train_subs,
            'tX_test' : tX_test_subs},

            {'regressor' : 'least_squares_SGD',
            'degrees' : 4,
            'params' : {'gamma': 5e-4, 'max_iters': 10000, 'seed': 0},
            'tX_train' : tX_train_subs,
            'tX_test' : tX_test_subs},

           {'regressor' : 'least_squares',
            'degrees' : 8,
            'params' : {},
            'tX_train' : tX_train_subs,
            'tX_test' : tX_test_subs},

           {'regressor' : 'ridge_regression',
            'degrees' : 8,
            'params' : {'lambda_': 1e-11},
            'tX_train' : tX_train_subs,
            'tX_test' : tX_test_subs},

            {'regressor' : 'logistic_regression',
            'degrees' : 4,
            'params' : {'gamma': 2e-6, 'max_iters': 500},
            'tX_train' : tX_train_subs,
            'tX_test' : tX_test_subs},

            {'regressor' : 'reg_logistic_regression',
            'degrees' : [0, 1, 2, 3, 4],
            'params' : {'lambda_': 1e-4, 'gamma': 2.5e-6, 'max_iters': 500},
            'tX_train' : tX_train_subs,
            'tX_test' : tX_test_subs},

            {'regressor' : 'optimized_regression',
             'base_regressor' : 'logistic_regression',
             'degrees' : 5,
             'params' : {'gamma': 1e-5, 'max_iters':500},
             'tX_train' : tX_train,
             'tX_test' : tX_test},

            {'regressor' : 'optimized_regression',
             'base_regressor' : 'reg_logistic_regression',
             'degrees' : 5,
             'params' : {'lambda_': 1e-9, 'gamma': 6.3e-6, 'max_iters': 500},
             'tX_train' : tX_train,
             'tX_test' : tX_test}]

for config in configs:

    if config['regressor'] != 'optimized_regression':

        # Augment the features with a polynomial basis (and standardize them)
        tX_test_poly = polynomial_basis(config['tX_train'], config['degrees'], True, config['tX_test'])
        tX_train_poly = polynomial_basis(config['tX_train'], config['degrees'], True, config['tX_train'])
        
        weights, _ = eval(config['regressor'])(y_train, tX_train_poly, **config['params'])
        y_pred = predict_labels(weights, tX_test_poly)
        create_csv_submission(ids_test, y_pred, 'data/submission_' + config['regressor'] + '.csv')
        y_pred_submission = np.genfromtxt('data/final_submission_' + config['regressor'] + '.csv', delimiter=",", skip_header=1, dtype=int, usecols=1)
        print(config['regressor'], 'found', sum(y_pred != y_pred_submission), 'different predictions from submissions.')

    else:

        # Do predictions, create submission-csv and compare to final submissions
        y_pred = optimized_regression(config['base_regressor'], config['tX_train'], y_train, config['tX_test'], config['degrees'], config['params'])
        create_csv_submission(ids_test, y_pred, 'data/submission_' + config['regressor'] + '_' + config['base_regressor'] + '.csv')
        y_pred_submission = np.genfromtxt('data/final_submission_' + config['regressor'] + '_' + config['base_regressor'] + '.csv', delimiter=",", skip_header=1, dtype=int, usecols=1)
        print(config['regressor'], 'with', config['base_regressor'], 'found', sum(y_pred != y_pred_submission), 'different predictions from submissions.')
"""
