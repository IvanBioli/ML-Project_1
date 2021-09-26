#%% Useful starting lines
import numpy as np
import matplotlib.pyplot as plt

# Loading the training data
from proj1_helpers import *
DATA_TRAIN_PATH = 'data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Standardizing the tX-data matrix
def standardize(tX : np.ndarray) -> np.ndarray:
    """
    Standardizes the columns in x to zero mean and unit variance.

    Parameters
    ----------
    tX : np.ndarray
        Array with the samples as rows and the features as columns.

    Returns
    -------
    tX_std : np.ndarray
        Standardized x with zero feature-means and unit feature-variance.
    """

    tX_std = (tX - np.mean(tX, axis=0)) / np.std(tX, axis=0)

    return tX_std

tX = standardize(tX)

# Fitting the regressor
from implementations import least_squares_GD

initial_w = np.ones(30, dtype=float)
weights, loss = least_squares_GD(y, tX, initial_w, max_iters=100, gamma=0.1)

# Obtaining the test-data
DATA_TEST_PATH = 'data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Creating the predictions
OUTPUT_PATH = 'data/submission.csv'
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
