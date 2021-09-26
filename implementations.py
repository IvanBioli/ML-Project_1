# -*- coding: utf-8 -*-
"""
Implementations
***************

Collection of machine learning algorithms for the project 1.
"""

import numpy as np

def least_squares_GD(y, tX, initial_w, max_iters=100, gamma=0.1):
    """
    Gradient descent algorithm for mean squared error (MSE) loss function.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    initial_w : np.ndarray
        Vector with initial weights to start the iteration from.
    max_iters : int, default=100
        Maximum number of iterations.
    gamma : float, default=0.1
        Scaling factor for the gradient subtraction.

    Returns
    -------
    w : np.ndarray
        Vector containing the final weights.
    loss : float
        Mean squared error loss function evaluated with the final weights.

    References
    ----------
    [1] M. Jaggi, and M. E. Khan, "Optimization", Machine Learning (CS-433),
        pp. 6-7, September 23, 2021.

    Usage
    -----
    >>> y, tX, _ = load_csv_data([TRAINING_DATA_PATH])
    >>> initial_w = np.zeros(tX.shape[1], dtype=float)
    >>> w, loss = least_squares_GD(y, tX, initial_w, max_iters=100, gamma=0.1)

    """

    w = initial_w

    for n_iter in range(max_iters):

        # Updating weights with scaled negative gradient
        w = w - gamma * (- tX.T @ (y - tX @ w) / (2 * len(y)))

    # Computing loss for the final weights
    loss = (y - tX @ w).T @ (y - tX @ w) / (2 * len(y))

    return w, loss