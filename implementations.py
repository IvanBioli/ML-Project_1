# -*- coding: utf-8 -*-
"""
Implementations
***************

Collection of machine learning algorithms for the project 1.
"""

import numpy as np


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
        e = y - tX @ w
        w = w + tX.T @ e * (gamma / len(y))         #only one multiplication scalar-matrix
                                                    #TODO: check if with y too big this creates problems since (gamma/len(y)) could be really small (even under epsilon machine)

    # Computing loss for the final weights (MSE)
    e = y - tX @ w
    loss = np.mean(e**2) / 2

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
    N = len(y)

    # Sampling random sequence of indices
    rand_ind = np.random.choice(np.arange(len(y)), max_iters, replace=False)

    for i in range(max_iters):
        y_rand = y[rand_ind[i]]
        tX_rand = tX[rand_ind[i]]
        # Updating weights with scaled negative gradient
        w = w + (gamma * (y_rand - np.inner(tX_rand, w)) / N) * tX_rand         #use np.inner instead of np.dot better only one matrix-vector multiplication
                                                                                #TODO: check if with y too big this creates problems since (gamma/len(y)) could be really small (even under epsilon machine)
        # TODO: find a way to stop the algorithm efficiently

    # Computing loss for the final weights (MSE)
    e = y - tX @ w                                                    #Compute the error e only once
    loss = np.mean(e**2) / 2

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

    # Computing loss for the final weights (MSE)
    e = y - tX @ w
    loss = e.T @ e / (2 * len(y))

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
    perturbation = lambda_ * 2 * tX.shape[0] * np.identity(tX.shape[1])
    w = np.linalg.inv(tX.T @ tX + perturbation) @ tX.T @ y

    # Computing loss for the final weights (MSE)
    e = y - tX @ w
    loss = e.T @ e / (2 * len(y))

    return w, loss


def logistic_regression(y, tX, initial_w, max_iters=100, gamma=0.1):
    """
    Gradient descent regressor with logistic loss function.

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
    [4] M. Jaggi, and M. E. Khan, "Logistic Regression",
        Machine Learning (CS-433), pp. 2-12, October XX, 2021.

    Usage
    -----
    >>> y, tX, _ = load_csv_data([TRAINING_DATA_PATH])
    >>> initial_w = np.zeros(tX.shape[1], dtype=float)
    >>> max_iters = 100
    >>> gamma = 0.1
    >>> w, loss = logistic_regression(y, tX, initial_w, max_iters, gamma)

    """

    w = initial_w

    # *_clipped quantities are there to ensure that np.exp(x) won't overflow,
    # because for x > 710 => exp(x) > 1.8e+308 > np.finfo('d').max.
    # They also ensure that np.log(x) won't be passed any non-positive values.
    for i in range(max_iters):

        # Evaluating the logistic function
        tXw_clipped = np.clip(tX @ w, -709*np.ones(len(y)), 709*np.ones(len(y)))
        sigma = 1 / (1 + np.exp( - tXw_clipped))

        # Updating weights with scaled negative gradient
        w = w - gamma * tX.T @ (sigma - y)

    # Computing loss for the final weights
    tXw_clipped = np.clip(tX @ w, -709*np.ones(len(y)), 709*np.ones(len(y)))
    sigma = 1 / (1 + np.exp( - tXw_clipped))

    sigma_clipped1 = np.clip(sigma, 1e-10, 709*np.ones(len(sigma)))
    sigma_clipped2 = np.clip(sigma, -709*np.ones(len(sigma)), -1e-10)
    loss = y.T @ np.log(sigma_clipped1) + (1.0 - y).T @ np.log(1.0 - np.exp(sigma_clipped2))

    return w, loss


def reg_logistic_regression(y, tX, lambda_, initial_w, max_iters=100, gamma=0.1):
    """
    Gradient descent regressor with (ridge) regularized logistic loss function.

    Parameters
    ----------
    y : np.ndarray
        Vector with the labels.
    tX : np.ndarray
        Array with the samples as rows and the features as columns.
    lambda_ : float
        Regularization parameter.
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
    [5] M. Jaggi, and M. E. Khan, "Logistic Regression",
        Machine Learning (CS-433), pp. 16-17, October XX, 2021.

    Usage
    -----
    >>> y, tX, _ = load_csv_data([TRAINING_DATA_PATH])
    >>> lambda_ = 0.1
    >>> initial_w = np.zeros(tX.shape[1], dtype=float)
    >>> max_iters = 100
    >>> gamma = 0.1
    >>> w, loss = logistic_regression(y, tX, initial_w, max_iters, gamma)

    """

    w = initial_w

    # *_clipped quantities are there to ensure that np.exp(x) won't overflow,
    # because for x > 710 => exp(x) > 1.8e+308 > np.finfo('d').max.
    # They also ensure that np.log(x) won't be passed any non-positive values.
    for i in range(max_iters):

        # Evaluating the logistic function
        tXw_clipped = np.clip(tX @ w, -709*np.ones(len(y)), 709*np.ones(len(y)))
        sigma = 1 / (1 + np.exp( - tXw_clipped))

        # Updating weights with scaled negative gradient (with penalty term [5])
        w = w - gamma * (tX.T @ (sigma - y) + lambda_ * w)

    # Computing loss for the final weights
    tXw_clipped = np.clip(tX @ w, -709*np.ones(len(y)), 709*np.ones(len(y)))
    sigma = 1 / (1 + np.exp( - tXw_clipped))

    sigma_clipped1 = np.clip(sigma, 1e-10, 709*np.ones(len(sigma)))
    sigma_clipped2 = np.clip(sigma, -709*np.ones(len(sigma)), -1e-10)
    loss = y.T @ np.log(sigma_clipped1) + (1.0 - y).T @ np.log(1.0 - np.exp(sigma_clipped2)) + lambda_ * w.T @ w / 2

    return w, loss
