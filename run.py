# -*- coding: utf-8 -*-
"""
Run
***

Run this file in order to obtain the predictions of our regressors.
"""

# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt

# Loading and standardizing the training data
from proj1_helpers import *
DATA_TRAIN_PATH = 'data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
tX = standardize(tX)


#%% Fitting the regressors
from implementations import *

initial_w = np.ones(30, dtype=float)
gamma = 0.1
max_iters = 100
lambda_ = 0.1

weights, loss = least_squares_GD(y, tX, initial_w, max_iters, gamma)
weights, loss = least_squares_SGD(y, tX, initial_w, max_iters, gamma)
weights, loss = least_squares(y, tX)
weights, loss = ridge_regression(y, tX, lambda_)
# weights, loss = logistic_regression(y, tX, initial_w, max_iters, gamma)
# weights, loss = reg_logistic_regression(y, tX, lambda_, initial_w, max_iters, gamma)

# Obtaining the test-data
DATA_TEST_PATH = 'data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Creating the predictions
OUTPUT_PATH = 'data/submission.csv'
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

# %%
