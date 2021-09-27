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
    >>> max_iters = 100
    >>> gamma = 0.1
    >>> w, loss = least_squares_GD(y, tX, initial_w, max_iters, gamma)

    """

    w = initial_w

    for n_iter in range(max_iters):

        # Updating weights with scaled negative gradient
        w = w - gamma * (- tX.T @ (y - tX @ w) / (2 * len(y)))

    # Computing loss for the final weights
    loss = (y - tX @ w).T @ (y - tX @ w) / (2 * len(y))

    return w, loss

def least_squares_SGD(y, tX, initial_w, max_iters=100, gamma=0.1):
    """
    Stochastic gradient descent algorithm for mean squared error (MSE) loss.

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
    [2] M. Jaggi, and M. E. Khan, "Optimization", Machine Learning (CS-433),
        pp. 8-10, September 23, 2021.

    Usage
    -----
    >>> y, tX, _ = load_csv_data([TRAINING_DATA_PATH])
    >>> initial_w = np.zeros(tX.shape[1], dtype=float)
    >>> max_iters = 100
    >>> gamma = 0.1
    >>> w, loss = least_squares_GD(y, tX, initial_w, max_iters, gamma)
    
    """

    w = initial_w

    for n_iter in range(max_iters):

        # Sampling a random index
        rand_ind = np.random.choice(np.arange(len(y)))

        y_rand = y[rand_ind]
        tX_rand = tX[rand_ind]

        # Updating weights with scaled negative gradient
        w = w - gamma * np.dot(- tX_rand.T, y_rand - tX_rand @ w) / 2

    # Computing loss for the final weights
    loss = np.dot((y_rand - tX_rand @ w).T, y_rand - tX_rand @ w) / 2

    return w, loss

def least_squares(y, tX):
    """
    Exact analytical solution for the weights using the normal equation.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.

    Returns
    -------
    w : np.ndarray
        Vector containing the final weights.
    loss : float
        Mean squared error loss function evaluated with the final weights.

    References
    ----------
    [2] M. Jaggi, and M. E. Khan, "Least Squares", Machine Learning (CS-433),
        p. 7, September XX, 2021.

    Usage
    -----
    >>> y, tX, _ = load_csv_data([TRAINING_DATA_PATH])
    >>> w, loss = least_squares(y, tX)

    """

    # Computing the exact analytical weights using the formula provided in [2]
    w = np.linalg.inv(tX.T @ tX) @ tX.T @ y

    # Computing loss for the final weights
    loss = (y - tX @ w).T @ (y - tX @ w) / (2 * len(y))

    return w, loss

def ridge_regression(y, tX, lambda_=0.1):
    """
    Exact analytical solution for the weights using the ridge-regularized
    normal equations.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    lambda_ : float, default=0.1
        Ridge-regularization parameter.

    Returns
    -------
    w : np.ndarray
        Vector containing the final weights.
    loss : float
        Mean squared error loss function evaluated with the final weights.

    References
    ----------
    [3] M. Jaggi, and M. E. Khan, "Regularization: Ridge Regression and Lasso",
        Machine Learning (CS-433), p. 3, September XX, 2021.

    Usage
    -----
    >>> y, tX, _ = load_csv_data([TRAINING_DATA_PATH])
    >>> lambda = 0.1
    >>> w, loss = least_squares(y, tX, lambda)

    """

    # Computing the exact analytical weights using the formula provided in [2]
    w = np.linalg.inv(tX.T @ tX + lambda_ / (2 * tX.shape[1]) * np.identity(tX.shape[1])) @ tX.T @ y

    # Computing loss for the final weights
    loss = (y - tX @ w).T @ (y - tX @ w) / (2 * len(y))

    return w, loss
